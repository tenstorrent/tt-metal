// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm, dependent on the RMSNORM define.
 * For rmsnorm we receive E(x**2) and compute the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 */

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_math.hpp"

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(2);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t do_beta = get_compile_time_arg_val(4);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(5) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(7) == 1;

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
    uint32_t cb_times_gamma_out = cb_out;
    if constexpr (do_gamma and do_beta) {
        cb_times_gamma_out = tt::CBIndex::c_13;
    }

    binary_op_init_common(cb_inp, cb_inp, cb_var);

    cb_wait_front(cb_reduce, 1);  // comes from the reader
    cb_wait_front(cb_eps, 1);     // comes from the reader
    // gamma / beta are persistent across all NCHt iterations — hoist the waits
    // out of the loop (raw kernel waited them per-iteration; idempotent since
    // they are never popped inside the loop).
    if constexpr (do_gamma) {
        cb_wait_front(cb_gamma, Wt);
    }
    if constexpr (do_beta) {
        cb_wait_front(cb_beta, Wt);
    }

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
         * RMSNorm reduces sum(x**2) directly into cb_var for rsqrt computation.
         * Uses auto-batched STREAMING mode - library handles CB lifecycle.
         */
        compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW>(
            cb_stats, cb_reduce, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols));

        // Stage 2 — cb_recip_sqrt_var = rsqrt(cb_var + cb_eps)
        // cb_var streams (produced by reduce); cb_eps is persistent (waited before NCHt loop).
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::NONE,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_var,
            cb_eps,
            cb_recip_sqrt_var,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            compute_kernel_lib::sfpu_chain(
                compute_kernel_lib::Rsqrt<
                    LEGACY_RSQRT ? compute_kernel_lib::Legacy::On : compute_kernel_lib::Legacy::Off>{}));

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */

        uint32_t normed_output_cb = cb_x_normed;
        if constexpr (!do_gamma) {
            normed_output_cb = cb_out;
        }

        // Stage 3 — normed = cb_norm_x_input * cb_recip_sqrt_var (COL bcast)
        // cb_recip_sqrt_var is produced by stage 2 and consumed within this single
        // iteration. WaitUpfrontPopAtEnd encapsulates wait-once-pop-at-end lifecycle.
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_norm_x_input, cb_recip_sqrt_var, normed_output_cb, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        if constexpr (do_gamma) {
            // Stage 4 — cb_times_gamma_out = cb_x_normed * cb_gamma (ROW bcast)
            // cb_gamma is persistent across NCHt (waited once outside the loop below,
            // popped at the end of kernel_main); within the helper it's NoWaitNoPop.
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_x_normed, cb_gamma, cb_times_gamma_out, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

            if constexpr (do_beta) {
                // Stage 5 — cb_out = cb_times_gamma_out + cb_beta (ROW bcast)
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::ROW,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                    cb_times_gamma_out, cb_beta, cb_out, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
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
