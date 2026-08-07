// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs, per device statistics, and gamma, beta, epsilon from interleaved dram.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/debug/assert.h"
#include <tt-metalium/constants.hpp>

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t stats_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(3);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(4);
    constexpr uint32_t transformation_mat_cb = get_compile_time_arg_val(5);
    constexpr uint32_t rope_cos_cb = get_compile_time_arg_val(6);
    constexpr uint32_t rope_sin_cb = get_compile_time_arg_val(7);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(8);
    constexpr uint32_t block_size = get_compile_time_arg_val(9);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(10);
    constexpr uint32_t reduce_factor = get_compile_time_arg_val(11);
    constexpr uint32_t epsilon_value = get_compile_time_arg_val(12);
    constexpr uint32_t has_weight = get_compile_time_arg_val(13);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(14);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(15);
    constexpr auto input_args = TensorAccessorArgs<16>();
    constexpr auto stats_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto transformation_mat_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto rope_cos_args = TensorAccessorArgs<transformation_mat_args.next_compile_time_args_offset()>();
    constexpr auto rope_sin_args = TensorAccessorArgs<rope_cos_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t transformation_mat_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_cos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rope_sin_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);

    // ublocks size defined in tiles
    const uint32_t input_tile_bytes = get_tile_size(input_cb);
    const uint32_t stats_tile_bytes = get_tile_size(stats_cb);
    const uint32_t weight_tile_bytes = get_tile_size(weight_cb);
    const uint32_t transformation_mat_tile_bytes = get_tile_size(transformation_mat_cb);
    const uint32_t rope_cos_tile_bytes = get_tile_size(rope_cos_cb);
    const uint32_t rope_sin_tile_bytes = get_tile_size(rope_sin_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto stats_accessor = TensorAccessor(stats_args, stats_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    const auto transformation_mat_accessor = TensorAccessor(transformation_mat_args, transformation_mat_addr);
    const auto rope_cos_accessor = TensorAccessor(rope_cos_args, rope_cos_addr);
    const auto rope_sin_accessor = TensorAccessor(rope_sin_args, rope_sin_addr);

    Noc noc;
    CircularBuffer cb_input(input_cb);
    CircularBuffer cb_stats(stats_cb);
    CircularBuffer cb_weight(weight_cb);
    CircularBuffer cb_transformation_mat(transformation_mat_cb);
    CircularBuffer cb_rope_cos(rope_cos_cb);
    CircularBuffer cb_rope_sin(rope_sin_cb);

    /**
     * Op asserts that weight input is bf16.
     * We can calculate the bytes in a face-row and face for usage when reading the weight.
     */
    constexpr uint32_t bf16_datum_size_bytes = 2;
    constexpr uint32_t face_row_bytes = tt::constants::FACE_WIDTH * bf16_datum_size_bytes;
    constexpr uint32_t face_bytes = tt::constants::FACE_HW * bf16_datum_size_bytes;

    // Generate constant tiles for layernorm compute
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        reduce_scalar_cb,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        reduce_factor>();
    generate_bcast_col_scalar(CircularBuffer(epsilon_cb), epsilon_value);

    if constexpr (fuse_rope) {
        // Read the single-tile transformation matrix for ROPE.
        cb_transformation_mat.reserve_back(1);
        noc.async_read(
            transformation_mat_accessor,
            cb_transformation_mat,
            transformation_mat_tile_bytes,
            {.page_id = 0},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_transformation_mat.push_back(1);
    }

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t stats_tile_idx = tile_row * stats_tiles_cols;
        // Read stats tiles
        cb_stats.reserve_back(stats_tiles_cols);
        uint32_t stats_wr_offset = 0;
        for (uint32_t col_tile = 0; col_tile < stats_tiles_cols; col_tile++) {
            noc.async_read(
                stats_accessor,
                cb_stats,
                stats_tile_bytes,
                {.page_id = stats_tile_idx},
                {.offset_bytes = stats_wr_offset});
            stats_wr_offset += stats_tile_bytes;
            stats_tile_idx++;
        }
        noc.async_read_barrier();
        cb_stats.push_back(stats_tiles_cols);

        // read input tiles
        uint32_t input_tile_idx = tile_row * num_tile_cols;
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_input.reserve_back(block_size);
            uint32_t input_wr_offset = 0;

            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                noc.async_read(
                    input_accessor,
                    cb_input,
                    input_tile_bytes,
                    {.page_id = input_tile_idx},
                    {.offset_bytes = input_wr_offset});
                input_wr_offset += input_tile_bytes;
                input_tile_idx++;
            }
            noc.async_read_barrier();
            cb_input.push_back(block_size);

            if constexpr (has_weight) {
                if (tile_row == tile_row_start) {
                    cb_weight.reserve_back(block_size);
                    uint32_t weight_wr_offset = 0;
                    for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                        // Rather than read a full tile containing sparse data,
                        // just read the first row of the tile from the faces.
                        noc.async_read(
                            weight_accessor,
                            cb_weight,
                            face_row_bytes /*one face row*/,
                            {.page_id = col_tile + i, .offset_bytes = 0},
                            {.offset_bytes = weight_wr_offset});
                        noc.async_read(
                            weight_accessor,
                            cb_weight,
                            face_row_bytes /*one face row*/,
                            {.page_id = col_tile + i, .offset_bytes = face_bytes},
                            {.offset_bytes = weight_wr_offset + face_bytes});
                        weight_wr_offset += weight_tile_bytes;
                    }
                    noc.async_read_barrier();
                    cb_weight.push_back(block_size);
                }
            }
            if constexpr (fuse_rope) {
                if (col_tile == 0) {
                    /**
                     * When processing the first column of a specific row, read the sin/cos inputs for rope.
                     */
                    uint32_t rope_tile_start_idx = tile_row * head_dim_tiles;
                    cb_rope_cos.reserve_back(head_dim_tiles);
                    uint32_t rope_cos_wr_offset = 0;
                    for (uint32_t i = 0; i < head_dim_tiles; i++) {
                        noc.async_read(
                            rope_cos_accessor,
                            cb_rope_cos,
                            rope_cos_tile_bytes,
                            {.page_id = rope_tile_start_idx + i},
                            {.offset_bytes = rope_cos_wr_offset});
                        rope_cos_wr_offset += rope_cos_tile_bytes;
                    }
                    noc.async_read_barrier();
                    cb_rope_cos.push_back(head_dim_tiles);

                    cb_rope_sin.reserve_back(head_dim_tiles);
                    uint32_t rope_sin_wr_offset = 0;
                    for (uint32_t i = 0; i < head_dim_tiles; i++) {
                        noc.async_read(
                            rope_sin_accessor,
                            cb_rope_sin,
                            rope_sin_tile_bytes,
                            {.page_id = rope_tile_start_idx + i},
                            {.offset_bytes = rope_sin_wr_offset});
                        rope_sin_wr_offset += rope_sin_tile_bytes;
                    }
                    noc.async_read_barrier();
                    cb_rope_sin.push_back(head_dim_tiles);
                }
            }
        }
    }
}
