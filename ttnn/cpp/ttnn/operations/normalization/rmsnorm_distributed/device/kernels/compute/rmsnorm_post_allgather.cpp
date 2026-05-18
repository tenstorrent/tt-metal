// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm, dependent on the RMSNORM define.
 * For rmsnorm we receive E(x**2) and compute the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(2);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t do_beta = get_compile_time_arg_val(4);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(5) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(7) == 1;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;

    constexpr uint32_t cb_eps = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_5;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;          // var + epsilon (or E(x**2) + epsilon)
    constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_x_normed = tt::CBIndex::c_12;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)

    constexpr uint32_t cb_var = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)
    constexpr uint32_t cb_norm_x_input = cb_inp;

    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    constexpr uint32_t cb_times_gamma_out = (do_gamma && do_beta) ? tt::CBIndex::c_13 : cb_out;

    binary_op_init_common(cb_inp, cb_inp, cb_var);

    cb_wait_front(cb_reduce, 1);  // comes from the reader
    cb_wait_front(cb_eps, 1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into cb_var for rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles CB lifecycle.
         */
        compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(
            cb_stats, cb_reduce, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols));

        /*
         * 1/sqrt(var + eps)
         * PARTIAL migration: BinaryFpu(Add, cb_var, cb_eps) + Rsqrt + PackTile(cb_recip_sqrt_var).
         */
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_var,
                    cb_eps,
                    BinaryFpuOp::Add,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>{},
                Rsqrt<Approx::Exact, LEGACY_RSQRT ? Legacy::On : Legacy::Off, Dst::D0>{},
                PackTile<
                    cb_recip_sqrt_var,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */

        constexpr uint32_t normed_output_cb = do_gamma ? cb_x_normed : cb_out;

        // Three eltwise stages via eltwise_chain. Each stage is a single chain
        // call over the full Wt with BlockSize=blk; the chain owns all of:
        //   - A-side wait_upfront / pop_at_end on the streaming input CB,
        //   - B-side index walk (FirstTile for cb_recip_sqrt_var, BlockIter for
        //     cb_gamma / cb_beta which already span all Wt tiles upfront),
        //   - pack-side reserve_upfront / push_at_end on the output CB,
        //   - entry-time srca/srcb + pack reconfig (fold-driven, compile-time
        //     elided when prev == curr).
        // The C++ kernel only sets up the once-per-NCHt wait on the B operands
        // (cb_recip_sqrt_var per ncht; cb_gamma / cb_beta once upfront for all
        // NCHt — they are reused across rows and the trailing cb_pop_front
        // after the NCHt loop releases them).

        /*
         * x * 1/sqrt(stdev)
         */
        cb_wait_front(cb_recip_sqrt_var, 1);
        {
            using namespace compute_kernel_lib;
            eltwise_chain<blk>(
                Wt,
                BinaryFpu<
                    cb_norm_x_input,
                    cb_recip_sqrt_var,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Col,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitUpfrontPopAtEnd,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::BlockIter,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                PackTile<
                    normed_output_cb,
                    Dst::D0,
                    PackTilePolicy::UpfrontReservePushAtEnd,
                    PackTileIndexMode::BlockIter,
                    PackTileReconfig::Output>{});
        }
        cb_pop_front(cb_recip_sqrt_var, 1);

        if constexpr (do_gamma) {
            /*
             * x_normed * gamma   (cb_gamma walks BlockIter — index = 0..Wt-1)
             */
            cb_wait_front(cb_gamma, Wt);  // gamma is reused across NCHt — wait once per row
            using namespace compute_kernel_lib;
            eltwise_chain<blk>(
                Wt,
                BinaryFpu<
                    cb_x_normed,
                    cb_gamma,
                    BinaryFpuOp::Mul,
                    BroadcastDim::Row,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitUpfrontPopAtEnd,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::BlockIter,
                    Dst::D0,
                    CbIndexMode::BlockIter>{},
                PackTile<
                    cb_times_gamma_out,
                    Dst::D0,
                    PackTilePolicy::UpfrontReservePushAtEnd,
                    PackTileIndexMode::BlockIter,
                    PackTileReconfig::Output>{});

            if constexpr (do_beta) {
                /*
                 * (x_normed * gamma) + beta   (cb_beta walks BlockIter — 0..Wt-1)
                 */
                cb_wait_front(cb_beta, Wt);
                eltwise_chain<blk>(
                    Wt,
                    BinaryFpu<
                        cb_times_gamma_out,
                        cb_beta,
                        BinaryFpuOp::Add,
                        BroadcastDim::Row,
                        BinaryDataFormatReconfig::Input,
                        CopyTilePolicy::WaitUpfrontPopAtEnd,
                        CopyTilePolicy::NoWaitNoPop,
                        CbIndexMode::BlockIter,
                        Dst::D0,
                        CbIndexMode::BlockIter>{},
                    PackTile<
                        cb_out,
                        Dst::D0,
                        PackTilePolicy::UpfrontReservePushAtEnd,
                        PackTileIndexMode::BlockIter,
                        PackTileReconfig::Output>{});
            }
        }
    }
    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_reduce, 1);
    if constexpr (do_gamma) {
        cb_pop_front(cb_gamma, Wt);
    }
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
