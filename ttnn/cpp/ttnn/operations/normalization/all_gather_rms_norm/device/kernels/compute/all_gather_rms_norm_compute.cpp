// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single fused compute kernel for all_gather_rms_norm.
//
// A Tensix core runs exactly one compute kernel, so the whole fused pipeline lives here:
//   1. pre-reduce: E[x^2] over the local-width (Wt) slice of x,
//   2. (ring_size > 1) all-gather the per-device partials and combine into the global E[x^2] -- TODO,
//   3. post-normalize: x / sqrt(E[x^2] + eps) * gamma + beta.
//
// The input row (Wt tiles) is read once into cb_inp and kept resident: it is used for x^2 AND for the
// normalize step, then popped at the end of the row.
//
// Math ported from the generic distributed kernels:
//   rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp  (E[x^2])
//   rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp (normalize + gamma/beta)
//
// STATUS: the ring_size == 1 path (single device, no fabric) is complete. The ring_size > 1 stats
// all-gather is left as a TODO (see below) — it needs the writer's fabric exchange + cb_gathered_stats.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

namespace NAMESPACE {
void MAIN {
    // Compile-time args (order must match all_gather_rms_norm_program_factory.cpp: compute_ct_args).
    constexpr uint32_t cb_inp = get_compile_time_arg_val(0);
    constexpr uint32_t cb_reduce = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(5);
    constexpr uint32_t cb_var = get_compile_time_arg_val(6);
    constexpr uint32_t cb_recip_sqrt_var = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x_normed = get_compile_time_arg_val(8);
    constexpr uint32_t cb_gamma_out = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t cb_local_stats = get_compile_time_arg_val(11);
    constexpr uint32_t cb_gathered_stats = get_compile_time_arg_val(12);
    constexpr uint32_t Wt = get_compile_time_arg_val(13);
    constexpr uint32_t blk = get_compile_time_arg_val(14);
    constexpr uint32_t ring_size = get_compile_time_arg_val(15);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(16);
    constexpr uint32_t do_beta = get_compile_time_arg_val(17);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(18) == 1;

    constexpr bool LEGACY_RSQRT = false;

    // Runtime arg: number of tile-rows this worker processes.
    const uint32_t NCHt = get_arg_val<uint32_t>(0);

    (void)cb_local_stats;
    (void)cb_gathered_stats;
    (void)FLOAT32_DTYPE;

    uint32_t cb_times_gamma_out = cb_out;
    if constexpr (do_gamma and do_beta) {
        cb_times_gamma_out = cb_gamma_out;
    }

    binary_op_init_common(cb_inp, cb_inp, cb_var);

    cb_wait_front(cb_reduce, 1);  // reader-provided AVG reduce scalar (1/reduce_factor)
    cb_wait_front(cb_eps, 1);     // reader-provided epsilon

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Bring the whole input row (Wt tiles) resident; keep it for both x^2 and the normalize step.
        cb_wait_front(cb_inp, Wt);

        // --- x^2 over the local width ---
        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);
        mul_tiles_init(cb_inp, cb_inp);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_reserve_back(cb_x2, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(cb_inp, cb_inp, wt + wtr, wt + wtr, wtr);
                pack_tile(wtr, cb_x2, wt + wtr);
            }
            REL();
            cb_push_back(cb_x2, blk);
        }

        // --- E[x^2] ---
        if constexpr (ring_size == 1) {
            // Single device: AVG-reduce x^2 over the (full) width directly into cb_var.
            compute_kernel_lib::
                reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                    cb_x2, cb_reduce, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        } else {
            // TODO(fabric): reduce x^2 -> cb_local_stats (per-device partial), let the writer all-gather it
            // into cb_gathered_stats (ring_size tiles), then reduce those into cb_var. Placeholder below
            // keeps the single-device math so the kernel still compiles.
            compute_kernel_lib::
                reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                    cb_x2, cb_reduce, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        }

        // --- 1/sqrt(E[x^2] + eps) ---
        cb_wait_front(cb_var, 1);
        cb_reserve_back(cb_recip_sqrt_var, 1);
        reconfig_data_format(cb_var, cb_eps);
        pack_reconfig_data_format(cb_recip_sqrt_var);
        add_tiles_init(cb_var, cb_eps);
        ACQ();
        add_tiles(cb_var, cb_eps, 0, 0, 0);
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(0);
        pack_tile(0, cb_recip_sqrt_var);
        REL();
        cb_push_back(cb_recip_sqrt_var, 1);
        cb_pop_front(cb_var, 1);

        // --- normalize: x * 1/sqrt(E[x^2] + eps)  (broadcast the per-row scalar over the Wt columns) ---
        uint32_t normed_output_cb = cb_x_normed;
        if constexpr (!do_gamma) {
            normed_output_cb = cb_out;
        }
        reconfig_data_format(cb_inp, cb_recip_sqrt_var);
        pack_reconfig_data_format(normed_output_cb);
        mul_bcast_cols_init_short(cb_inp, cb_recip_sqrt_var);
        cb_wait_front(cb_recip_sqrt_var, 1);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_reserve_back(normed_output_cb, blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles_bcast_cols(cb_inp, cb_recip_sqrt_var, wt + wtr, 0, wtr);
                pack_tile(wtr, normed_output_cb);
            }
            REL();
            cb_push_back(normed_output_cb, blk);
        }
        cb_pop_front(cb_recip_sqrt_var, 1);
        cb_pop_front(cb_inp, Wt);  // done with the input row

        if constexpr (do_gamma) {
            // x_normed * gamma  (gamma broadcast over rows)
            reconfig_data_format(cb_x_normed, cb_gamma);
            pack_reconfig_data_format(cb_times_gamma_out);
            cb_wait_front(cb_gamma, Wt);
            mul_bcast_rows_init_short(cb_x_normed, cb_gamma);
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                cb_wait_front(cb_x_normed, blk);
                cb_reserve_back(cb_times_gamma_out, blk);
                ACQ();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    mul_tiles_bcast_rows(cb_x_normed, cb_gamma, wtr, wt + wtr, wtr);
                    pack_tile(wtr, cb_times_gamma_out);
                }
                REL();
                cb_push_back(cb_times_gamma_out, blk);
                cb_pop_front(cb_x_normed, blk);
            }

            if constexpr (do_beta) {
                // x_normed * gamma + beta  (beta broadcast over rows)
                reconfig_data_format(cb_times_gamma_out, cb_beta);
                pack_reconfig_data_format(cb_out);
                cb_wait_front(cb_beta, Wt);
                add_bcast_rows_init_short(cb_times_gamma_out, cb_beta);
                for (uint32_t wt = 0; wt < Wt; wt += blk) {
                    cb_wait_front(cb_times_gamma_out, blk);
                    cb_reserve_back(cb_out, blk);
                    ACQ();
                    for (uint32_t wtr = 0; wtr < blk; wtr++) {
                        add_tiles_bcast_rows(cb_times_gamma_out, cb_beta, wtr, wt + wtr, wtr);
                        pack_tile(wtr, cb_out);
                    }
                    REL();
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
}  // namespace NAMESPACE
