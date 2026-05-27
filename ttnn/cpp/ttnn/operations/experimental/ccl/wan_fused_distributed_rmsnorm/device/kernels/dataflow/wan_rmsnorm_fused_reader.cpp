// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Reader for the fused Wan2.2 distributed RMSNorm op.
 *
 * Streams the core's tile-row slice of (input, weight, rope_cos, rope_sin)
 * from DRAM into local CBs, block-by-block — matches the existing
 * rms_post_allgather_reader streaming model so compute can start consuming
 * the first chunk while the reader continues filling.
 *
 * Differences from the existing post-allgather reader:
 *   - We do NOT read stats from DRAM; stats come from the compute kernel's
 *     own pre phase via stats_local_cb, are forwarded by the AG kernel, and
 *     are delivered to compute via stats_gathered_cb.
 *   - We generate TWO reduce scalars: SUM (for pre phase) and AVG (for post).
 *   - Optional weight / RoPE / trans_mat loading is preserved verbatim.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(1);
    constexpr uint32_t reduce_scalar_sum_cb = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_scalar_avg_cb = get_compile_time_arg_val(3);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(4);
    constexpr uint32_t transformation_mat_cb = get_compile_time_arg_val(5);
    constexpr uint32_t rope_cos_cb = get_compile_time_arg_val(6);
    constexpr uint32_t rope_sin_cb = get_compile_time_arg_val(7);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(8);
    constexpr uint32_t block_size = get_compile_time_arg_val(9);
    constexpr uint32_t reduce_factor = get_compile_time_arg_val(10);  // == stats_tiles_cols
    constexpr uint32_t epsilon_value = get_compile_time_arg_val(11);
    constexpr uint32_t has_weight = get_compile_time_arg_val(12);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(13);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t chunk_size_rows = get_compile_time_arg_val(15);
    constexpr uint32_t per_head_rope = get_compile_time_arg_val(16);
    constexpr uint32_t rope_seqlen_tiles = get_compile_time_arg_val(17);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(18);
    constexpr uint32_t has_bias = get_compile_time_arg_val(19);
    constexpr auto input_args = TensorAccessorArgs<20>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto transformation_mat_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto rope_cos_args = TensorAccessorArgs<transformation_mat_args.next_compile_time_args_offset()>();
    constexpr auto rope_sin_args = TensorAccessorArgs<rope_cos_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t transformation_mat_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_cos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_sin_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t input_tile_bytes = get_tile_size(input_cb);
    const uint32_t weight_tile_bytes = get_tile_size(weight_cb);
    const uint32_t bias_tile_bytes = get_tile_size(bias_cb);
    const uint32_t rope_cos_tile_bytes = get_tile_size(rope_cos_cb);
    const uint32_t rope_sin_tile_bytes = get_tile_size(rope_sin_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr);
    const auto transformation_mat_accessor = TensorAccessor(transformation_mat_args, transformation_mat_addr);
    const auto rope_cos_accessor = TensorAccessor(rope_cos_args, rope_cos_addr);
    const auto rope_sin_accessor = TensorAccessor(rope_sin_args, rope_sin_addr);

    // Generate reduce scalars (SUM for pre uses 1.0, AVG for post uses 1/H_full)
    // and the eps tile.
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        reduce_scalar_sum_cb,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        reduce_scalar_avg_cb,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        reduce_factor>();
    generate_bcast_col_scalar(epsilon_cb, epsilon_value);

    if constexpr (fuse_rope) {
        cb_reserve_back(transformation_mat_cb, 1);
        uint32_t transformation_mat_wr_ptr = get_write_ptr(transformation_mat_cb);
        noc_async_read_tile(0, transformation_mat_accessor, transformation_mat_wr_ptr);
        noc_async_read_barrier();
        cb_push_back(transformation_mat_cb, 1);
    }

    // Weight + bias are consumed in the POST phase (sub-phases 2 / 2.5) which
    // only start after chunk 0's AG completes. So both reads can be deferred
    // until chunk 0's input rows are all pushed — the latency then hides
    // behind chunk 0's pre compute + fabric mcast + fabric wait. Issued in
    // `block_size`-sized pushes so the compute kernel can consume cumulatively.
    bool weight_pushed = (has_weight == 0);
    bool bias_pushed = (has_bias == 0);

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t input_tile_idx = tile_row * num_tile_cols;
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            // Clamp the block to the actual remaining tiles in the row so we
            // never reserve more than the CB can hold (which would deadlock
            // when num_tile_cols < block_size).
            const uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            cb_reserve_back(input_cb, tiles_in_block);
            uint32_t input_wr_ptr = get_write_ptr(input_cb);

            for (uint32_t i = 0; i < tiles_in_block; i++) {
                noc_async_read_tile(input_tile_idx, input_accessor, input_wr_ptr);
                input_wr_ptr += input_tile_bytes;
                input_tile_idx++;
            }
            noc_async_read_barrier();
            cb_push_back(input_cb, tiles_in_block);

            if constexpr (fuse_rope) {
                if (col_tile == 0) {
                    // Per-head RoPE: cos/sin shape [B, num_heads, N, head_dim].
                    // Push all heads' head_dim_tiles tiles for this tile_row,
                    // packed contiguously. Tile idx for head h, tile t, col c:
                    //   h * rope_seqlen_tiles * head_dim_tiles + t * head_dim_tiles + c
                    //
                    // Broadcast RoPE (per_head_rope=0): cos/sin shape
                    // [B, 1, N, head_dim], indexed by t * head_dim_tiles + c.
                    if constexpr (per_head_rope != 0) {
                        const uint32_t num_heads_per_device = num_tile_cols / head_dim_tiles;
                        const uint32_t total_tiles = num_heads_per_device * head_dim_tiles;
                        cb_reserve_back(rope_cos_cb, total_tiles);
                        cb_reserve_back(rope_sin_cb, total_tiles);
                        uint32_t rope_cos_wr_ptr = get_write_ptr(rope_cos_cb);
                        uint32_t rope_sin_wr_ptr = get_write_ptr(rope_sin_cb);
                        for (uint32_t h = 0; h < num_heads_per_device; h++) {
                            const uint32_t head_base =
                                h * rope_seqlen_tiles * head_dim_tiles + tile_row * head_dim_tiles;
                            for (uint32_t i = 0; i < head_dim_tiles; i++) {
                                noc_async_read_tile(head_base + i, rope_cos_accessor, rope_cos_wr_ptr);
                                rope_cos_wr_ptr += rope_cos_tile_bytes;
                                noc_async_read_tile(head_base + i, rope_sin_accessor, rope_sin_wr_ptr);
                                rope_sin_wr_ptr += rope_sin_tile_bytes;
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(rope_cos_cb, total_tiles);
                        cb_push_back(rope_sin_cb, total_tiles);
                    } else {
                        uint32_t rope_tile_start_idx = tile_row * head_dim_tiles;
                        cb_reserve_back(rope_cos_cb, head_dim_tiles);
                        cb_reserve_back(rope_sin_cb, head_dim_tiles);
                        uint32_t rope_cos_wr_ptr = get_write_ptr(rope_cos_cb);
                        uint32_t rope_sin_wr_ptr = get_write_ptr(rope_sin_cb);
                        for (uint32_t i = 0; i < head_dim_tiles; i++) {
                            noc_async_read_tile(rope_tile_start_idx + i, rope_cos_accessor, rope_cos_wr_ptr);
                            rope_cos_wr_ptr += rope_cos_tile_bytes;
                            noc_async_read_tile(rope_tile_start_idx + i, rope_sin_accessor, rope_sin_wr_ptr);
                            rope_sin_wr_ptr += rope_sin_tile_bytes;
                        }
                        noc_async_read_barrier();
                        cb_push_back(rope_cos_cb, head_dim_tiles);
                        cb_push_back(rope_sin_cb, head_dim_tiles);
                    }
                }
            }
        }

        // After chunk 0's rows are pushed (or at end-of-worker if the worker
        // has fewer rows than chunk_size_rows), issue the weight + bias reads.
        // Weight comes first because the compute kernel consumes it first
        // (sub-phase 2: x_rms * weight). Bias is consumed in sub-phase 2.5
        // (+ bias) immediately after.
        const uint32_t rows_pushed = tile_row + 1 - tile_row_start;
        const bool first_chunk_done = (rows_pushed >= chunk_size_rows);
        const bool last_row = (tile_row + 1 == tile_row_end);
        const bool should_issue_side_inputs = first_chunk_done || last_row;
        if (!weight_pushed && should_issue_side_inputs) {
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_reserve_back(weight_cb, tiles_in_block);
                uint32_t weight_wr_ptr = get_write_ptr(weight_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    noc_async_read_tile(col_tile + i, weight_accessor, weight_wr_ptr);
                    weight_wr_ptr += weight_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(weight_cb, tiles_in_block);
            }
            weight_pushed = true;
        }
        if (!bias_pushed && should_issue_side_inputs) {
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_reserve_back(bias_cb, tiles_in_block);
                uint32_t bias_wr_ptr = get_write_ptr(bias_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    noc_async_read_tile(col_tile + i, bias_accessor, bias_wr_ptr);
                    bias_wr_ptr += bias_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(bias_cb, tiles_in_block);
            }
            bias_pushed = true;
        }
    }
}
