// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram.
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/assert.h"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_val = get_compile_time_arg_val(4);
    constexpr auto input_args = TensorAccessorArgs<5>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);  // Source address in dram
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    const uint32_t input_tile_bytes = get_tile_size(input_cb);

    const auto input_accessor = TensorAccessor(input_args, input_addr, input_tile_bytes);

    // Generate constant tiles for reduce scalar
    generate_reduce_scaler(reduce_scalar_cb, scalar_val);

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t input_tile_idx = tile_row * num_tile_cols;
        // read input tiles
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_reserve_back(input_cb, block_size);
            uint32_t input_wr_ptr = get_write_ptr(input_cb);
            for (uint32_t r = 0; r < block_size && col_tile + r < num_tile_cols; r++) {
                noc_async_read_tile(input_tile_idx, input_accessor, input_wr_ptr);
                input_wr_ptr += input_tile_bytes;
                input_tile_idx++;
            }
            noc_async_read_barrier();
            cb_push_back(input_cb, block_size);
        }
    }
}
