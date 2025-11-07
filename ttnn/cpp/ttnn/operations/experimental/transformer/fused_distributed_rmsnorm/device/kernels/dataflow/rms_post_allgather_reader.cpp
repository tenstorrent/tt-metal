// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs, per device statistics, and gamma, beta, epsilon from interleaved dram.
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/assert.h"
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
    constexpr uint32_t scalar_value = get_compile_time_arg_val(11);
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

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);
    const auto stats_accessor = TensorAccessor(stats_args, stats_addr, stats_tile_bytes);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, weight_tile_bytes);
    const auto transformation_mat_accessor =
        TensorAccessor(transformation_mat_args, transformation_mat_addr, transformation_mat_tile_bytes);
    const auto rope_cos_accessor = TensorAccessor(rope_cos_args, rope_cos_addr, rope_cos_tile_bytes);
    const auto rope_sin_accessor = TensorAccessor(rope_sin_args, rope_sin_addr, rope_sin_tile_bytes);

    /**
     * Op asserts that weight input is bf16.
     * We can calculate the bytes in a face-row and face for usage when reading the weight.
     */
    constexpr uint32_t bf16_datum_size_bytes = 2;
    constexpr uint32_t face_row_bytes = tt::constants::FACE_WIDTH * bf16_datum_size_bytes;
    constexpr uint32_t face_bytes = tt::constants::FACE_HW * bf16_datum_size_bytes;

    // Generate constant tiles for layernorm compute
    generate_reduce_scaler(reduce_scalar_cb, scalar_value);
    generate_bcast_col_scalar(epsilon_cb, epsilon_value);

    if constexpr (fuse_rope) {
        // Read the single-tile transformation matrix for ROPE.
        cb_reserve_back(transformation_mat_cb, 1);
        uint32_t transformation_mat_wr_ptr = get_write_ptr(transformation_mat_cb);
        noc_async_read_tile(0, transformation_mat_accessor, transformation_mat_wr_ptr);
        noc_async_read_barrier();
        cb_push_back(transformation_mat_cb, 1);
    }

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t stats_tile_idx = tile_row * stats_tiles_cols;
        // Read stats tiles
        cb_reserve_back(stats_cb, stats_tiles_cols);
        uint32_t stats_wr_ptr = get_write_ptr(stats_cb);
        for (uint32_t col_tile = 0; col_tile < stats_tiles_cols; col_tile++) {
            noc_async_read_tile(stats_tile_idx, stats_accessor, stats_wr_ptr);
            stats_wr_ptr += stats_tile_bytes;
            stats_tile_idx++;
        }
        noc_async_read_barrier();
        cb_push_back(stats_cb, stats_tiles_cols);

        // read input tiles
        uint32_t input_tile_idx = tile_row * num_tile_cols;
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_reserve_back(input_cb, block_size);
            uint32_t input_wr_ptr = get_write_ptr(input_cb);

            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                noc_async_read_tile(input_tile_idx, input_accessor, input_wr_ptr);
                input_wr_ptr += input_tile_bytes;
                input_tile_idx++;
            }
            noc_async_read_barrier();
            cb_push_back(input_cb, block_size);

            if constexpr (has_weight) {
                if (tile_row == tile_row_start) {
                    cb_reserve_back(weight_cb, block_size);
                    uint32_t weight_wr_ptr = get_write_ptr(weight_cb);
                    for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                        uint64_t weight_noc_addr = get_noc_addr(col_tile + i, weight_accessor);

                        // Rather than read a full tile containing sparse data,
                        // just read the first row of the tile from the faces.
                        noc_async_read(weight_noc_addr, weight_wr_ptr, face_row_bytes /*one face row*/);
                        noc_async_read(
                            weight_noc_addr + face_bytes, weight_wr_ptr + face_bytes, face_row_bytes /*one face row*/);
                        weight_wr_ptr += weight_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(weight_cb, block_size);
                }
            }
            if constexpr (fuse_rope) {
                if (col_tile == 0) {
                    /**
                     * When processing the first column of a specific row, read the sin/cos inputs for rope.
                     */
                    uint32_t rope_tile_start_idx = tile_row * head_dim_tiles;
                    cb_reserve_back(rope_cos_cb, head_dim_tiles);
                    uint32_t rope_cos_wr_ptr = get_write_ptr(rope_cos_cb);
                    for (uint32_t i = 0; i < head_dim_tiles; i++) {
                        noc_async_read_tile(rope_tile_start_idx + i, rope_cos_accessor, rope_cos_wr_ptr);
                        rope_cos_wr_ptr += rope_cos_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(rope_cos_cb, head_dim_tiles);

                    cb_reserve_back(rope_sin_cb, head_dim_tiles);
                    uint32_t rope_sin_wr_ptr = get_write_ptr(rope_sin_cb);
                    for (uint32_t i = 0; i < head_dim_tiles; i++) {
                        noc_async_read_tile(rope_tile_start_idx + i, rope_sin_accessor, rope_sin_wr_ptr);
                        rope_sin_wr_ptr += rope_sin_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(rope_sin_cb, head_dim_tiles);
                }
            }
        }
    }
}
