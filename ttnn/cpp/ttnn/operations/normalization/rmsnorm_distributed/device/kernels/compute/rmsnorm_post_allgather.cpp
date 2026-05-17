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

        // Per-block streaming via eltwise_chain. Three nested stages:
        //   1) x * recip(stdev)  — col-bcast B (cb_recip_sqrt_var) pinned at 0
        //   2) * gamma            — row-bcast B (cb_gamma) walks wt+wtr globally
        //   3) + beta             — row-bcast B (cb_beta) walks wt+wtr globally
        //
        // Stages 2 and 3 use the new BlockIterOffset index mode (chain returns
        // b_tile_idx_ + j); caller passes wt as the per-outer-iter base via ctor.
        // All operands NoWaitNoPop (caller owns wait/pop outside chain) and the
        // pack uses NoReserveNoPush (caller owns reserve+push) — the C++ outer
        // loop preserves the original per-blk wait/reserve/pop/push lifecycle.
        cb_wait_front(cb_recip_sqrt_var, 1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_wait_front(cb_norm_x_input, blk);
            cb_reserve_back(normed_output_cb, blk);
            {
                using namespace compute_kernel_lib;
                eltwise_chain(
                    blk,
                    BinaryFpu<
                        cb_norm_x_input,
                        cb_recip_sqrt_var,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Col,
                        BinaryDataFormatReconfig::Input,
                        CopyTilePolicy::NoWaitNoPop,
                        CopyTilePolicy::NoWaitNoPop,
                        CbIndexMode::BlockIter,
                        Dst::D0,
                        CbIndexMode::FirstTile>{},
                    PackTile<
                        normed_output_cb,
                        Dst::D0,
                        PackTilePolicy::NoReserveNoPush,
                        PackTileIndexMode::BlockIter,
                        PackTileReconfig::Output>{});
            }
            cb_push_back(normed_output_cb, blk);
            cb_pop_front(cb_norm_x_input, blk);
        }
        cb_pop_front(cb_recip_sqrt_var, 1);

        if constexpr (do_gamma) {
            /*
             * x_normed * gamma
             */
            cb_wait_front(cb_gamma, Wt);
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                cb_wait_front(cb_x_normed, blk);
                cb_reserve_back(cb_times_gamma_out, blk);
                {
                    using namespace compute_kernel_lib;
                    eltwise_chain(
                        blk,
                        BinaryFpu<
                            cb_x_normed,
                            cb_gamma,
                            BinaryFpuOp::Mul,
                            BroadcastDim::Row,
                            BinaryDataFormatReconfig::Input,
                            CopyTilePolicy::NoWaitNoPop,
                            CopyTilePolicy::NoWaitNoPop,
                            CbIndexMode::BlockIter,
                            Dst::D0,
                            CbIndexMode::BlockIterOffset>{/*a_tile_idx=*/0, /*b_tile_idx=*/wt},
                        PackTile<
                            cb_times_gamma_out,
                            Dst::D0,
                            PackTilePolicy::NoReserveNoPush,
                            PackTileIndexMode::BlockIter,
                            PackTileReconfig::Output>{});
                }
                cb_push_back(cb_times_gamma_out, blk);
                cb_pop_front(cb_x_normed, blk);
            }

            if constexpr (do_beta) {
                /*
                 * x_normed * gamma + beta
                 */
                cb_wait_front(cb_beta, Wt);
                for (uint32_t wt = 0; wt < Wt; wt += blk) {
                    cb_wait_front(cb_times_gamma_out, blk);
                    cb_reserve_back(cb_out, blk);
                    {
                        using namespace compute_kernel_lib;
                        eltwise_chain(
                            blk,
                            BinaryFpu<
                                cb_times_gamma_out,
                                cb_beta,
                                BinaryFpuOp::Add,
                                BroadcastDim::Row,
                                BinaryDataFormatReconfig::Input,
                                CopyTilePolicy::NoWaitNoPop,
                                CopyTilePolicy::NoWaitNoPop,
                                CbIndexMode::BlockIter,
                                Dst::D0,
                                CbIndexMode::BlockIterOffset>{/*a_tile_idx=*/0, /*b_tile_idx=*/wt},
                            PackTile<
                                cb_out,
                                Dst::D0,
                                PackTilePolicy::NoReserveNoPush,
                                PackTileIndexMode::BlockIter,
                                PackTileReconfig::Output>{});
                    }
                    cb_push_back(cb_out, blk);
                    cb_pop_front(cb_times_gamma_out, blk);
                }
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
