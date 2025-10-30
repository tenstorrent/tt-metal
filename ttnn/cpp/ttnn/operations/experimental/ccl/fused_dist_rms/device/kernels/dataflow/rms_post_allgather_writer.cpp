// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes tiles from the output buffer to interleaved dram.
 */

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);

    constexpr auto output_args = TensorAccessorArgs<3>();
    const uint32_t tile_bytes = get_tile_size(output_cb);

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t tile_id = tile_row * num_tile_cols;
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_wait_front(output_cb, block_size);
            uint32_t output_read_ptr = get_read_ptr(output_cb);
            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                noc_async_write_tile(tile_id, output_accessor, output_read_ptr);
                output_read_ptr += tile_bytes;
                tile_id++;
            }
            noc_async_writes_flushed();
            cb_pop_front(output_cb, block_size);
        }
    }
    noc_async_write_barrier();
}
