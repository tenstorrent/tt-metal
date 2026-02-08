// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * LayerNorm-only Welford post-allgather.
 * Expects stats with two TILE columns per device (E(x**2), E(x)), applies LN with optional gamma/beta.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h"

void kernel_main() {
    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    constexpr uint32_t cb_eps = tt::CBIndex::c_4;
    constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_5;   // [mean, var]
    constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_6;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_intermediate = tt::CBIndex::c_7;    // intermediate result
    constexpr uint32_t cb_out = tt::CBIndex::c_8;

    constexpr uint32_t stats_tile_stride = 2;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_devices = get_compile_time_arg_val(3);

    constexpr uint32_t do_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t do_beta = get_compile_time_arg_val(5);
    constexpr uint32_t gamma_is_batched = get_compile_time_arg_val(6);
    constexpr uint32_t beta_is_batched = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);

    constexpr uint32_t Wt_round_up_block_sizes = get_compile_time_arg_val(9);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);

    binary_op_init_common(cb_inp, cb_inp, cb_stats_reduced);

    cb_wait_front(cb_eps, 1);  // broadcast epsilon is ready

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Calculate global tile row and batch index
        uint32_t global_tile_row = tile_row_start + tile_row;
        uint32_t batch_idx = global_tile_row / Ht;
        // Combine per-device stats into mean/variance
        norm::kernel_util::compute::combine_welford_partials(
            cb_stats,
            cb_stats_reduced,
            num_devices,
            [&](uint32_t) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        cb_push_back(cb_stats_reduced, stats_tile_stride);
        cb_wait_front(cb_stats_reduced, stats_tile_stride);

        // Compute 1/sqrt(var + eps) into cb_recip_sqrt_var
        cb_reserve_back(cb_recip_sqrt_var, 1);
        reconfig_data_format(cb_stats_reduced, cb_eps);
        pack_reconfig_data_format(cb_recip_sqrt_var);

        add_tiles_init(cb_stats_reduced, cb_eps);
        rsqrt_tile_init<true>();
        tile_regs_acquire();
        tile_regs_wait();
        // stats_reduced tile 1 holds variance (after combine_welford_partials)
        add_tiles(cb_stats_reduced, cb_eps, 1, 0, 0);
        rsqrt_tile<true>(0);
        pack_tile(0, cb_recip_sqrt_var);
        tile_regs_commit();
        tile_regs_release();
        cb_push_back(cb_recip_sqrt_var, 1);

        // Process tiles across width in blocks
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            // 1) x_minus_mean
            reconfig_data_format(cb_inp, cb_stats_reduced);
            pack_reconfig_data_format(cb_intermediate);
            sub_bcast_cols_init_short(cb_inp, cb_stats_reduced);
            cb_wait_front(cb_inp, block_size);
            cb_reserve_back(cb_intermediate, block_size);
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size; i++) {
                sub_tiles_bcast_cols(cb_inp, cb_stats_reduced, i, 0, i);
                pack_tile(i, cb_intermediate);
            }
            tile_regs_commit();
            tile_regs_release();
            cb_pop_front(cb_inp, block_size);
            cb_push_back(cb_intermediate, block_size);

            // 2) normalize: (x-mean) * inv_std
            constexpr uint32_t norm_target_cb = (do_gamma || do_beta) ? cb_intermediate : cb_out;
            reconfig_data_format(cb_intermediate, cb_recip_sqrt_var);
            pack_reconfig_data_format(norm_target_cb);
            mul_bcast_cols_init_short(cb_intermediate, cb_recip_sqrt_var);
            cb_wait_front(cb_intermediate, block_size);
            cb_wait_front(cb_recip_sqrt_var, 1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_size; i++) {
                mul_tiles_bcast_cols(cb_intermediate, cb_recip_sqrt_var, i, 0, i);
            }
            tile_regs_commit();

            // Note that compute and pack are separated because it's possible that
            // norm_target_cb == cb_intermediate (in the case of no gamma/beta), so this
            // must be able to support in-place operations.
            cb_pop_front(cb_intermediate, block_size);
            cb_reserve_back(norm_target_cb, block_size);
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size; i++) {
                pack_tile(i, norm_target_cb);
            }
            tile_regs_release();
            cb_push_back(norm_target_cb, block_size);

            // 3) optional gamma
            if constexpr (do_gamma) {
                constexpr uint32_t gamma_out_cb = do_beta ? cb_intermediate : cb_out;
                reconfig_data_format(norm_target_cb, cb_gamma);
                pack_reconfig_data_format(gamma_out_cb);
                mul_bcast_rows_init_short(norm_target_cb, cb_gamma);

                cb_wait_front(cb_gamma, col_tile + block_size);
                cb_wait_front(norm_target_cb, block_size);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size; i++) {
                    mul_tiles_bcast_rows(norm_target_cb, cb_gamma, i, col_tile + i, i);
                }
                tile_regs_commit();

                cb_pop_front(norm_target_cb, block_size);
                cb_reserve_back(gamma_out_cb, block_size);

                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    pack_tile(i, gamma_out_cb);
                }
                tile_regs_release();

                cb_push_back(gamma_out_cb, block_size);
            }

            // 4) optional beta (only if gamma was provided)
            if constexpr (do_beta) {
                // Input is always in cb_intermediate, output is always cb_out
                reconfig_data_format(cb_intermediate, cb_beta);
                pack_reconfig_data_format(cb_out);
                add_bcast_rows_init_short(cb_intermediate, cb_beta);

                cb_wait_front(cb_beta, col_tile + block_size);
                cb_wait_front(cb_intermediate, block_size);
                cb_reserve_back(cb_out, block_size);
                tile_regs_acquire();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    add_tiles_bcast_rows(cb_intermediate, cb_beta, i, col_tile + i, i);
                    pack_tile(i, cb_out);
                }
                tile_regs_commit();
                tile_regs_release();
                cb_pop_front(cb_intermediate, block_size);
                cb_push_back(cb_out, block_size);
            }
        }

        // free up per-row resources
        cb_pop_front(cb_stats_reduced, stats_tile_stride);
        cb_pop_front(cb_recip_sqrt_var, 1);

        // Check if next tile_row is in a different batch - if so, pop gamma/beta
        if (tile_row + 1 < num_tile_rows) {
            uint32_t next_global_tile_row = tile_row_start + tile_row + 1;
            uint32_t next_batch_idx = next_global_tile_row / Ht;
            if (next_batch_idx != batch_idx) {
                // Pop gamma/beta to prepare for next batch
                if constexpr (do_gamma && gamma_is_batched) {
                    cb_pop_front(cb_gamma, Wt_round_up_block_sizes);
                }
                if constexpr (do_beta && beta_is_batched) {
                    cb_pop_front(cb_beta, Wt_round_up_block_sizes);
                }
            }
        }
    }

    // Pop remaining gamma/beta at the end (if batched, only the last batch's data)
    if constexpr (do_gamma) {
        cb_pop_front(cb_gamma, Wt_round_up_block_sizes);
    }
    if constexpr (do_beta) {
        cb_pop_front(cb_beta, Wt_round_up_block_sizes);
    }

    cb_pop_front(cb_eps, 1);
}
