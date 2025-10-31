// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t stats_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(3);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(4);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(5);
    constexpr uint32_t block_size = get_compile_time_arg_val(6);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(7);
    constexpr uint32_t scalar_value = get_compile_time_arg_val(8);
    constexpr uint32_t epsilon_value = get_compile_time_arg_val(9);
    constexpr uint32_t has_weight = get_compile_time_arg_val(10);
    constexpr auto input_args = TensorAccessorArgs<11>();
    constexpr auto stats_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);  // Source address in dram
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    const uint32_t weight_addr = get_arg_val<uint32_t>(2);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(3);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(4);

    // ublocks size defined in tiles
    const uint32_t input_tile_bytes = get_tile_size(input_cb);
    const uint32_t stats_tile_bytes = get_tile_size(stats_cb);
    const uint32_t weight_tile_bytes = get_tile_size(weight_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);
    const auto stats_accessor = TensorAccessor(stats_args, stats_addr, stats_tile_bytes);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, weight_tile_bytes);

    // Generate constant tiles for layernorm compute
    generate_reduce_scaler(reduce_scalar_cb, scalar_value);
    generate_bcast_col_scalar(epsilon_cb, epsilon_value);

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
                        // noc_async_read_tile(col_tile, weight_accessor, weight_wr_ptr);
                        // weight_wr_ptr += weight_tile_bytes;

                        noc_async_read(weight_noc_addr, weight_wr_ptr, 16 * 2 /*one face row*/);
                        noc_async_read(weight_noc_addr + 512, weight_wr_ptr + 512, 16 * 2 /*one face row*/);
                        weight_wr_ptr += weight_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(weight_cb, block_size);
                }
            }
        }
    }
}
