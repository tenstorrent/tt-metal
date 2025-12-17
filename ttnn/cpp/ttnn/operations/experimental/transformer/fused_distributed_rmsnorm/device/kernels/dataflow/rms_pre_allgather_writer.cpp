// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes tiles from the output buffer to interleaved dram.
 */

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_tiles_per_row = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    const uint32_t tile_bytes = get_tile_size(output_cb);

    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t tile_id = tile_row * output_tiles_per_row;
        cb_wait_front(output_cb, output_tiles_per_row);
        uint32_t l1_read_addr = get_read_ptr(output_cb);
        for (uint32_t tile_col = 0; tile_col < output_tiles_per_row; tile_col++) {
            noc_async_write_tile(tile_id, output_accessor, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(output_cb, output_tiles_per_row);
    }
    noc_async_write_barrier();
}
