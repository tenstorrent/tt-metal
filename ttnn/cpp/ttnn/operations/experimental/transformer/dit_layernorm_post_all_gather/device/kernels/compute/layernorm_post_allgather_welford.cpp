// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * LayerNorm-only Welford post-allgather.
 * Expects stats with two TILE columns per device (E(x**2), E(x)), applies LN with optional gamma/beta.
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h"

void kernel_main() {
    constexpr uint32_t cb_inp_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma_id = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta_id = tt::CBIndex::c_3;
    constexpr uint32_t cb_eps_id = tt::CBIndex::c_4;
    constexpr uint32_t cb_stats_reduced_id = tt::CBIndex::c_5;   // [mean, var]
    constexpr uint32_t cb_recip_sqrt_var_id = tt::CBIndex::c_6;  // 1/sqrt(var+eps)
    constexpr uint32_t cb_intermediate_id = tt::CBIndex::c_7;    // intermediate result
    constexpr uint32_t cb_out_id = tt::CBIndex::c_8;

    CircularBuffer cb_inp(cb_inp_id);
    CircularBuffer cb_stats(cb_stats_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_beta(cb_beta_id);
    CircularBuffer cb_eps(cb_eps_id);
    CircularBuffer cb_stats_reduced(cb_stats_reduced_id);
    CircularBuffer cb_recip_sqrt_var(cb_recip_sqrt_var_id);
    CircularBuffer cb_intermediate(cb_intermediate_id);
    CircularBuffer cb_out(cb_out_id);

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

    compute_kernel_hw_startup(cb_inp_id, cb_inp_id, cb_stats_reduced_id);

    cb_eps.wait_front(1);  // broadcast epsilon is ready

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Calculate global tile row and batch index
        uint32_t global_tile_row = tile_row_start + tile_row;
        uint32_t batch_idx = global_tile_row / Ht;
        // Combine per-device stats into mean/variance
        norm::kernel_util::compute::combine_welford_partials(
            cb_stats,
            cb_stats_reduced,
            num_devices,
            [W](uint32_t) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        cb_stats_reduced.push_back(stats_tile_stride);
        cb_stats_reduced.wait_front(stats_tile_stride);

        // Compute 1/sqrt(var + eps) into cb_recip_sqrt_var_id
        cb_recip_sqrt_var.reserve_back(1);
        reconfig_data_format(cb_stats_reduced_id, cb_eps_id);
        pack_reconfig_data_format(cb_recip_sqrt_var_id);

        add_init(cb_stats_reduced_id, cb_eps_id);
        rsqrt_tile_init<true>();
        tile_regs_acquire();
        tile_regs_wait();
        // stats_reduced tile 1 holds variance (after combine_welford_partials)
        add_tiles(cb_stats_reduced_id, cb_eps_id, 1, 0, 0);
        rsqrt_tile<true>(0);
        pack_tile(0, cb_recip_sqrt_var_id);
        tile_regs_commit();
        tile_regs_release();
        cb_recip_sqrt_var.push_back(1);

        // Process tiles across width in blocks
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            // 1) x_minus_mean
            reconfig_data_format(cb_inp_id, cb_stats_reduced_id);
            pack_reconfig_data_format(cb_intermediate_id);
            sub_bcast_cols_init_short(cb_inp_id, cb_stats_reduced_id);
            cb_inp.wait_front(block_size);
            cb_intermediate.reserve_back(block_size);
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size; i++) {
                sub_tiles_bcast_cols(cb_inp_id, cb_stats_reduced_id, i, 0, i);
                pack_tile(i, cb_intermediate_id);
            }
            tile_regs_commit();
            tile_regs_release();
            cb_inp.pop_front(block_size);
            cb_intermediate.push_back(block_size);

            // 2) normalize: (x-mean) * inv_std
            constexpr uint32_t norm_target_cb = (do_gamma || do_beta) ? cb_intermediate_id : cb_out_id;
            CircularBuffer norm_target_cb_obj(norm_target_cb);
            reconfig_data_format(cb_intermediate_id, cb_recip_sqrt_var_id);
            pack_reconfig_data_format(norm_target_cb);
            mul_bcast_cols_init_short(cb_intermediate_id, cb_recip_sqrt_var_id);
            cb_intermediate.wait_front(block_size);
            cb_recip_sqrt_var.wait_front(1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_size; i++) {
                mul_tiles_bcast_cols(cb_intermediate_id, cb_recip_sqrt_var_id, i, 0, i);
            }
            tile_regs_commit();

            // Note that compute and pack are separated because it's possible that
            // norm_target_cb == cb_intermediate_id (in the case of no gamma/beta), so this
            // must be able to support in-place operations.
            cb_intermediate.pop_front(block_size);
            norm_target_cb_obj.reserve_back(block_size);
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size; i++) {
                pack_tile(i, norm_target_cb);
            }
            tile_regs_release();
            norm_target_cb_obj.push_back(block_size);

            // 3) optional gamma
            if constexpr (do_gamma) {
                constexpr uint32_t gamma_out_cb = do_beta ? cb_intermediate_id : cb_out_id;
                CircularBuffer gamma_out_cb_obj(gamma_out_cb);
                reconfig_data_format(norm_target_cb, cb_gamma_id);
                pack_reconfig_data_format(gamma_out_cb);
                mul_bcast_rows_init_short(norm_target_cb, cb_gamma_id);

                cb_gamma.wait_front(col_tile + block_size);
                norm_target_cb_obj.wait_front(block_size);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size; i++) {
                    mul_tiles_bcast_rows(norm_target_cb, cb_gamma_id, i, col_tile + i, i);
                }
                tile_regs_commit();

                norm_target_cb_obj.pop_front(block_size);
                gamma_out_cb_obj.reserve_back(block_size);

                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    pack_tile(i, gamma_out_cb);
                }
                tile_regs_release();

                gamma_out_cb_obj.push_back(block_size);
            }

            // 4) optional beta (only if gamma was provided)
            if constexpr (do_beta) {
                // Input is always in cb_intermediate_id, output is always cb_out_id
                reconfig_data_format(cb_intermediate_id, cb_beta_id);
                pack_reconfig_data_format(cb_out_id);
                add_bcast_rows_init_short(cb_intermediate_id, cb_beta_id);

                cb_beta.wait_front(col_tile + block_size);
                cb_intermediate.wait_front(block_size);
                cb_out.reserve_back(block_size);
                tile_regs_acquire();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    add_tiles_bcast_rows(cb_intermediate_id, cb_beta_id, i, col_tile + i, i);
                    pack_tile(i, cb_out_id);
                }
                tile_regs_commit();
                tile_regs_release();
                cb_intermediate.pop_front(block_size);
                cb_out.push_back(block_size);
            }
        }

        // free up per-row resources
        cb_stats_reduced.pop_front(stats_tile_stride);
        cb_recip_sqrt_var.pop_front(1);

        // Check if next tile_row is in a different batch - if so, pop gamma/beta
        if (tile_row + 1 < num_tile_rows) {
            uint32_t next_global_tile_row = tile_row_start + tile_row + 1;
            uint32_t next_batch_idx = next_global_tile_row / Ht;
            if (next_batch_idx != batch_idx) {
                // Pop gamma/beta to prepare for next batch
                if constexpr (do_gamma && gamma_is_batched) {
                    cb_gamma.pop_front(Wt_round_up_block_sizes);
                }
                if constexpr (do_beta && beta_is_batched) {
                    cb_beta.pop_front(Wt_round_up_block_sizes);
                }
            }
        }
    }

    // Pop remaining gamma/beta at the end (if batched, only the last batch's data)
    if constexpr (do_gamma) {
        cb_gamma.pop_front(Wt_round_up_block_sizes);
    }
    if constexpr (do_beta) {
        cb_beta.pop_front(Wt_round_up_block_sizes);
    }

    cb_eps.pop_front(1);
}
