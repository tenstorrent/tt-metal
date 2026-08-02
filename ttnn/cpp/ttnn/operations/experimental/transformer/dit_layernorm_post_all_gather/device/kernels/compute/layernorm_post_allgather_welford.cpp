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
    constexpr uint32_t dfb_inp_id = tt::CBIndex::c_0;
    constexpr uint32_t dfb_stats_id = tt::CBIndex::c_1;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_beta_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_4;
    constexpr uint32_t dfb_stats_reduced_id = tt::CBIndex::c_5;   // [mean, var]
    constexpr uint32_t dfb_recip_sqrt_var_id = tt::CBIndex::c_6;  // 1/sqrt(var+eps)
    constexpr uint32_t dfb_intermediate_id = tt::CBIndex::c_7;    // intermediate result
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_8;

    DataflowBuffer dfb_inp(dfb_inp_id);
    DataflowBuffer dfb_stats(dfb_stats_id);
    DataflowBuffer dfb_gamma(dfb_gamma_id);
    DataflowBuffer dfb_beta(dfb_beta_id);
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_stats_reduced(dfb_stats_reduced_id);
    DataflowBuffer dfb_recip_sqrt_var(dfb_recip_sqrt_var_id);
    DataflowBuffer dfb_intermediate(dfb_intermediate_id);
    DataflowBuffer dfb_out(dfb_out_id);

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

    binary_op_init_common(dfb_inp_id, dfb_inp_id, dfb_stats_reduced_id);

    dfb_eps.wait_front(1);  // broadcast epsilon is ready

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        // Calculate global tile row and batch index
        uint32_t global_tile_row = tile_row_start + tile_row;
        uint32_t batch_idx = global_tile_row / Ht;
        // Combine per-device stats into mean/variance
        norm::kernel_util::compute::combine_welford_partials(
            dfb_stats,
            dfb_stats_reduced,
            num_devices,
            [W](uint32_t) { return (static_cast<float>(W)); },
            norm::kernel_util::compute::RSqrtPolicy{false, 0});
        dfb_stats_reduced.push_back(stats_tile_stride);
        dfb_stats_reduced.wait_front(stats_tile_stride);

        // Compute 1/sqrt(var + eps) into cb_recip_sqrt_var_id
        dfb_recip_sqrt_var.reserve_back(1);
        reconfig_data_format(dfb_stats_reduced_id, dfb_eps_id);
        pack_reconfig_data_format(dfb_recip_sqrt_var_id);

        add_tiles_init(dfb_stats_reduced_id, dfb_eps_id);
        rsqrt_tile_init<true>();
        tile_regs_acquire();
        tile_regs_wait();
        // stats_reduced tile 1 holds variance (after combine_welford_partials)
        add_tiles(dfb_stats_reduced_id, dfb_eps_id, 1, 0, 0);
        rsqrt_tile<true>(0);
        pack_tile(0, dfb_recip_sqrt_var_id);
        tile_regs_commit();
        tile_regs_release();
        dfb_recip_sqrt_var.push_back(1);

        // Process tiles across width in blocks
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            // 1) x_minus_mean
            reconfig_data_format(dfb_inp_id, dfb_stats_reduced_id);
            pack_reconfig_data_format(dfb_intermediate_id);
            sub_bcast_cols_init_short(dfb_inp_id, dfb_stats_reduced_id);
            dfb_inp.wait_front(block_size);
            dfb_intermediate.reserve_back(block_size);
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size; i++) {
                sub_tiles_bcast_cols(dfb_inp_id, dfb_stats_reduced_id, i, 0, i);
                pack_tile(i, dfb_intermediate_id);
            }
            tile_regs_commit();
            tile_regs_release();
            dfb_inp.pop_front(block_size);
            dfb_intermediate.push_back(block_size);

            // 2) normalize: (x-mean) * inv_std
            constexpr uint32_t norm_target_dfb = (do_gamma || do_beta) ? dfb_intermediate_id : dfb_out_id;
            DataflowBuffer norm_target_dfb_obj(norm_target_dfb);
            reconfig_data_format(dfb_intermediate_id, dfb_recip_sqrt_var_id);
            pack_reconfig_data_format(norm_target_dfb);
            mul_bcast_cols_init_short(dfb_intermediate_id, dfb_recip_sqrt_var_id);
            dfb_intermediate.wait_front(block_size);
            dfb_recip_sqrt_var.wait_front(1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_size; i++) {
                mul_tiles_bcast_cols(dfb_intermediate_id, dfb_recip_sqrt_var_id, i, 0, i);
            }
            tile_regs_commit();

            // Note that compute and pack are separated because it's possible that
            // norm_target_cb == cb_intermediate_id (in the case of no gamma/beta), so this
            // must be able to support in-place operations.
            dfb_intermediate.pop_front(block_size);
            norm_target_dfb_obj.reserve_back(block_size);
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size; i++) {
                pack_tile(i, norm_target_dfb);
            }
            tile_regs_release();
            norm_target_dfb_obj.push_back(block_size);

            // 3) optional gamma
            if constexpr (do_gamma) {
                constexpr uint32_t gamma_out_dfb = do_beta ? dfb_intermediate_id : dfb_out_id;
                DataflowBuffer gamma_out_dfb_obj(gamma_out_dfb);
                reconfig_data_format(norm_target_dfb, dfb_gamma_id);
                pack_reconfig_data_format(gamma_out_dfb);
                mul_bcast_rows_init_short(norm_target_dfb, dfb_gamma_id);

                dfb_gamma.wait_front(col_tile + block_size);
                norm_target_dfb_obj.wait_front(block_size);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size; i++) {
                    mul_tiles_bcast_rows(norm_target_dfb, dfb_gamma_id, i, col_tile + i, i);
                }
                tile_regs_commit();

                norm_target_dfb_obj.pop_front(block_size);
                gamma_out_dfb_obj.reserve_back(block_size);

                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    pack_tile(i, gamma_out_dfb);
                }
                tile_regs_release();

                gamma_out_dfb_obj.push_back(block_size);
            }

            // 4) optional beta (only if gamma was provided)
            if constexpr (do_beta) {
                // Input is always in cb_intermediate_id, output is always cb_out_id
                reconfig_data_format(dfb_intermediate_id, dfb_beta_id);
                pack_reconfig_data_format(dfb_out_id);
                add_bcast_rows_init_short(dfb_intermediate_id, dfb_beta_id);

                dfb_beta.wait_front(col_tile + block_size);
                dfb_intermediate.wait_front(block_size);
                dfb_out.reserve_back(block_size);
                tile_regs_acquire();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size; i++) {
                    add_tiles_bcast_rows(dfb_intermediate_id, dfb_beta_id, i, col_tile + i, i);
                    pack_tile(i, dfb_out_id);
                }
                tile_regs_commit();
                tile_regs_release();
                dfb_intermediate.pop_front(block_size);
                dfb_out.push_back(block_size);
            }
        }

        // free up per-row resources
        dfb_stats_reduced.pop_front(stats_tile_stride);
        dfb_recip_sqrt_var.pop_front(1);

        // Check if next tile_row is in a different batch - if so, pop gamma/beta
        if (tile_row + 1 < num_tile_rows) {
            uint32_t next_global_tile_row = tile_row_start + tile_row + 1;
            uint32_t next_batch_idx = next_global_tile_row / Ht;
            if (next_batch_idx != batch_idx) {
                // Pop gamma/beta to prepare for next batch
                if constexpr (do_gamma && gamma_is_batched) {
                    dfb_gamma.pop_front(Wt_round_up_block_sizes);
                }
                if constexpr (do_beta && beta_is_batched) {
                    dfb_beta.pop_front(Wt_round_up_block_sizes);
                }
            }
        }
    }

    // Pop remaining gamma/beta at the end (if batched, only the last batch's data)
    if constexpr (do_gamma) {
        dfb_gamma.pop_front(Wt_round_up_block_sizes);
    }
    if constexpr (do_beta) {
        dfb_beta.pop_front(Wt_round_up_block_sizes);
    }

    dfb_eps.pop_front(1);
}
