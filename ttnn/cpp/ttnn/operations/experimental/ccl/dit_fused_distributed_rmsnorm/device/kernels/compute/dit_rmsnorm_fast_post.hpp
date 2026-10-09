// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fast-POST RMSNorm compute for the all-gather path with a broadcast weight and no bias /
 * RoPE / per-head norm (resident input, one row per iteration). Called from dit_rmsnorm_fused_compute.cpp
 * when CT arg 43 (post_mode) is non-zero; same CT-arg layout.
 *
 * PRE and the gathered-stats combine match dit_rmsnorm_fused_compute.cpp. POST:
 *   post_mode 1: y = x * weight is computed right after the local stat is pushed,
 *     while the stats all-gather is in flight; after the gather only y * (1/rms)
 *     remains.
 *   post_mode 2: seed order (x * 1/rms, then * weight) fused per block instead of
 *     sub-phase-major over the whole row, so the first output block is ready after
 *     one block of work and the writer drain overlaps the rest of POST.
 */

#pragma once

#include <cstdint>

template <uint32_t post_mode>
inline void dit_rmsnorm_fast_post_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t stats_local_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_scalar_sum_cb = get_compile_time_arg_val(4);
    constexpr uint32_t reduce_scalar_avg_cb = get_compile_time_arg_val(5);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(6);
    constexpr uint32_t reduce_result_cb = get_compile_time_arg_val(7);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(8);
    constexpr uint32_t pre_intermediate_cb = get_compile_time_arg_val(9);
    constexpr uint32_t output_cb = get_compile_time_arg_val(10);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(15);
    constexpr uint32_t block_size = get_compile_time_arg_val(16);
    // ring_size * column-split partials per row.
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(17);
    constexpr uint32_t stats_transposed_local_cb = get_compile_time_arg_val(22);
    constexpr uint32_t stats_transposed_gathered_cb = get_compile_time_arg_val(23);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(32);
    // Split drain: odd output blocks go to output2_cb (drained by the reader).
    constexpr uint32_t split_drain = get_compile_time_arg_val(44);
    constexpr uint32_t output2_cb = get_compile_time_arg_val(45);
    static_assert(
        post_mode == 0 || (stats_tiles_cols >= 2 && stats_tiles_cols % 2 == 0),
        "pairwise stats sum needs an even count");
    static_assert(post_mode == 1 || post_mode == 2, "unsupported post_mode");
    // num_tile_cols * stats_tiles_cols * 32 == full global width (column parts are equal).
    constexpr uint32_t recip_h_full_bits =
        __builtin_bit_cast(uint32_t, 1.0f / static_cast<float>(num_tile_cols * 32u * stats_tiles_cols));

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(input_cb, input_cb, pre_intermediate_cb);

    CircularBuffer cb_input(input_cb);
    CircularBuffer cb_stats_local(stats_local_cb);
    CircularBuffer cb_weight(weight_cb);
    CircularBuffer cb_reduce_scalar_sum(reduce_scalar_sum_cb);
    CircularBuffer cb_reduce_scalar_avg(reduce_scalar_avg_cb);
    CircularBuffer cb_epsilon(epsilon_cb);
    CircularBuffer cb_reduce_result(reduce_result_cb);
    CircularBuffer cb_intermediate(intermediate_cb);
    CircularBuffer cb_pre_intermediate(pre_intermediate_cb);
    CircularBuffer cb_output(output_cb);
    CircularBuffer cb_output2(output2_cb);
    CircularBuffer cb_stats_transposed_local(stats_transposed_local_cb);
    CircularBuffer cb_stats_transposed_gathered(stats_transposed_gathered_cb);

    cb_reduce_scalar_sum.wait_front(1);
    cb_reduce_scalar_avg.wait_front(1);
    cb_epsilon.wait_front(1);

    for (uint32_t row = 0; row < num_tile_rows; ++row) {
        // -------- PRE: sum(x**2) --------
        reconfig_data_format(input_cb, input_cb);
        pack_reconfig_data_format(pre_intermediate_cb);
        PACK((llk_pack_reconfig_l1_acc(0)));
        mul_init(input_cb, input_cb);
        cb_pre_intermediate.reserve_back(1);
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            cb_input.wait_front(col_tile + tiles_in_block);
            tile_regs_acquire();
            for (uint32_t i = 0; i < tiles_in_block; i++) {
                mul_tiles(input_cb, input_cb, col_tile + i, col_tile + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < tiles_in_block; i++) {
                pack_tile<true>(i, pre_intermediate_cb, 0);
                if (col_tile == 0 && i == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            tile_regs_release();
        }
        cb_pre_intermediate.push_back(1);
        PACK((llk_pack_reconfig_l1_acc(0)));
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, pre_intermediate_cb, reduce_scalar_sum_cb, stats_local_cb>(
                compute_kernel_lib::ReduceInputBlockShape::single());

        // Stat col 0 -> row 0 for the 128 B stick.
        transpose_init(stats_local_cb);
        pack_reconfig_data_format(stats_transposed_local_cb);
        cb_stats_local.wait_front(1);
        cb_stats_transposed_local.reserve_back(1);
        tile_regs_acquire();
        transpose_tile(stats_local_cb, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, stats_transposed_local_cb);
        tile_regs_release();
        cb_stats_transposed_local.push_back(1);
        cb_stats_local.pop_front(1);

        // -------- post_mode 1: x * weight while the gather is in flight --------
        if constexpr (post_mode == 1) {
            cb_weight.wait_front(num_tile_cols);
            reconfig_data_format(input_cb, weight_cb);
            pack_reconfig_data_format(intermediate_cb);
            mul_bcast_rows_init(input_cb, weight_cb);
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_intermediate.reserve_back(block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    mul_tiles_bcast_rows(input_cb, weight_cb, col_tile + i, col_tile + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    pack_tile(i, intermediate_cb);
                }
                tile_regs_release();
                cb_intermediate.push_back(block_size);
            }
        }

        // -------- combine gathered partial sums -> 1/rms in col 0 --------
        cb_stats_transposed_gathered.wait_front(stats_tiles_cols);
        reconfig_data_format(stats_transposed_gathered_cb, stats_transposed_gathered_cb);
        pack_reconfig_data_format(reduce_result_cb);
        tile_regs_acquire();
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(
            stats_transposed_gathered_cb, stats_transposed_gathered_cb, false);
        add_tiles(stats_transposed_gathered_cb, stats_transposed_gathered_cb, 0, 1, 0);
        binary_tiles_init<false, EltwiseBinaryType::ELWADD>(
            stats_transposed_gathered_cb, stats_transposed_gathered_cb, true);
        for (uint32_t k = 2; k < stats_tiles_cols; k += 2) {
            add_tiles(stats_transposed_gathered_cb, stats_transposed_gathered_cb, k, k + 1, 0);
        }
        transpose_dest_init<true>(stats_transposed_gathered_cb);
        transpose_dest<true>(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, recip_h_full_bits);
        add_unary_tile(0, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reduce_result.reserve_back(1);
        pack_tile(0, reduce_result_cb);
        cb_reduce_result.push_back(1);
        tile_regs_release();
        cb_stats_transposed_gathered.pop_front(stats_tiles_cols);
        cb_reduce_result.wait_front(1);

        if constexpr (post_mode == 1) {
            reconfig_data_format(intermediate_cb, reduce_result_cb);
            pack_reconfig_data_format(output_cb);
            mul_bcast_cols_init(intermediate_cb, reduce_result_cb);
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_intermediate.wait_front(block_size);
                const bool to_reader = split_drain && ((col_tile / block_size) & 1u);
                CircularBuffer& cb_out = to_reader ? cb_output2 : cb_output;
                const uint32_t out_cb_id = to_reader ? output2_cb : output_cb;
                cb_out.reserve_back(block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    mul_tiles_bcast_cols(intermediate_cb, reduce_result_cb, i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    pack_tile(i, out_cb_id);
                }
                tile_regs_release();
                cb_out.push_back(block_size);
                cb_intermediate.pop_front(block_size);
            }
        } else {
            // Block-fused seed order: per block, x * (1/rms) -> fp32 intermediate, then
            // immediately * weight -> output, so output blocks stream out while later
            // blocks are still being normalized.
            cb_weight.wait_front(num_tile_cols);
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                reconfig_data_format(input_cb, reduce_result_cb);
                pack_reconfig_data_format(intermediate_cb);
                mul_bcast_cols_init(input_cb, reduce_result_cb);
                cb_intermediate.reserve_back(block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    mul_tiles_bcast_cols(input_cb, reduce_result_cb, col_tile + i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    pack_tile(i, intermediate_cb);
                }
                tile_regs_release();
                cb_intermediate.push_back(block_size);

                reconfig_data_format(intermediate_cb, weight_cb);
                pack_reconfig_data_format(output_cb);
                mul_bcast_rows_init(intermediate_cb, weight_cb);
                cb_intermediate.wait_front(block_size);
                const bool to_reader = split_drain && ((col_tile / block_size) & 1u);
                CircularBuffer& cb_out = to_reader ? cb_output2 : cb_output;
                const uint32_t out_cb_id = to_reader ? output2_cb : output_cb;
                cb_out.reserve_back(block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    mul_tiles_bcast_rows(intermediate_cb, weight_cb, i, col_tile + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    pack_tile(i, out_cb_id);
                }
                tile_regs_release();
                cb_out.push_back(block_size);
                cb_intermediate.pop_front(block_size);
            }
        }
        cb_reduce_result.pop_front(1);
        cb_input.pop_front(num_tile_cols);
    }

    cb_reduce_scalar_sum.pop_front(1);
    cb_reduce_scalar_avg.pop_front(1);
    cb_epsilon.pop_front(1);
    cb_weight.pop_front(num_tile_cols);
}
