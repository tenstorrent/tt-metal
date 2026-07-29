// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes tiles from the output buffer to interleaved dram.
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_tiles_per_row = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    const uint32_t tile_bytes = get_tile_size(output_cb);

    const auto output_accessor = TensorAccessor(output_args, output_addr);

    Noc noc;
    CircularBuffer cb_output(output_cb);

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t tile_id = tile_row * output_tiles_per_row;
        cb_output.wait_front(output_tiles_per_row);
        uint32_t output_rd_offset = 0;
        for (uint32_t tile_col = 0; tile_col < output_tiles_per_row; tile_col++) {
            noc.async_write(
                cb_output, output_accessor, tile_bytes, {.offset_bytes = output_rd_offset}, {.page_id = tile_id});
            tile_id++;
            output_rd_offset += tile_bytes;
        }
        noc.async_writes_flushed();
        cb_output.pop_front(output_tiles_per_row);
    }
    noc.async_write_barrier();
}
