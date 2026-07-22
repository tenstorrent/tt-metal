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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

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
        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into cb_var for rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles CB lifecycle.
         */
        ckl::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, cb_stats, cb_reduce, cb_var>(
            ckl::ReduceInputBlockShape::row(stats_tiles_cols));

        /*
         * 1/sqrt(var + eps)
         * BinaryFpu(Add, cb_var, cb_eps) + Rsqrt + PackTile(cb_recip_sqrt_var).
         * cb_var: InputLifecycle::Streaming (per-tile wait/pop). cb_eps: InputLifecycle::CallerManaged (pre-waited
         * before NCHt loop, popped once after). cb_recip_sqrt_var: OutputLifecycle::Streaming.
         */
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(cb_var),
                ckl::input(cb_eps, ckl::InputLifecycle::CallerManaged),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_recip_sqrt_var)>{});

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         *
         * Three eltwise stages via eltwise_chain. Each stage is a single chain
         * call over the full Wt with BlockSize=blk; the chain owns all of:
         *   - A-side wait_upfront / pop_at_end on the streaming input CB,
         *   - B-side index walk (Scalar for cb_recip_sqrt_var, Block for
         *     cb_gamma / cb_beta which already span all Wt tiles upfront),
         *   - pack-side reserve_upfront / push_at_end on the output CB,
         *   - entry-time srca/srcb + pack reconfig.
         * The chain now owns the B-side CB edges too: cb_recip_sqrt_var is Bulk
         * (wait+pop per call); cb_gamma / cb_beta are HeldBulk (chain waits Wt per
         * row, no pop) — they stay resident across rows, released by the trailing
         * cb_pop_front after the NCHt loop.
         */
        constexpr uint32_t normed_output_cb = do_gamma ? cb_x_normed : cb_out;

        ckl::mul<
            ckl::input(cb_norm_x_input, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
            ckl::input(cb_recip_sqrt_var, ckl::InputLifecycle::Bulk),
            ckl::output(normed_output_cb, ckl::OutputLifecycle::Bulk),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(Wt, /*block_size=*/blk));

        if constexpr (do_gamma) {
            /*
             * x_normed * gamma   (cb_gamma walks Block — index = 0..Wt-1)
             */
            ckl::mul<
                ckl::input(cb_x_normed, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(cb_gamma, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
                ckl::output(cb_times_gamma_out, ckl::OutputLifecycle::Bulk),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::tiles(Wt, /*block_size=*/blk));

            if constexpr (do_beta) {
                /*
                 * (x_normed * gamma) + beta   (cb_beta walks Block — 0..Wt-1)
                 */
                ckl::add<
                    ckl::input(cb_times_gamma_out, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(cb_beta, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
                    ckl::output(cb_out, ckl::OutputLifecycle::Bulk),
                    ckl::BroadcastDim::Row>(ckl::EltwiseShape::tiles(Wt, /*block_size=*/blk));
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
