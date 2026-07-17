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

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_heads = get_compile_time_arg_val(4);
    constexpr uint32_t num_tile_rows = get_compile_time_arg_val(5);

    constexpr auto output_args = TensorAccessorArgs<6>();
    const uint32_t tile_bytes = get_tile_size(output_cb);

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, output_addr);

    Noc noc;
    CircularBuffer cb_output(output_cb);

    constexpr uint32_t head_stride = num_tile_rows * head_dim_tiles;

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t head_idx = 0;
        uint32_t tile_idx_in_head = 0;

        uint32_t tile_id = tile_row * head_dim_tiles;

        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_output.wait_front(block_size);
            uint32_t output_rd_offset = 0;
            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                noc.async_write(
                    cb_output, output_accessor, tile_bytes, {.offset_bytes = output_rd_offset}, {.page_id = tile_id});
                output_rd_offset += tile_bytes;
                tile_id++;
                tile_idx_in_head++;
                if (tile_idx_in_head == head_dim_tiles) {
                    tile_idx_in_head = 0;
                    head_idx++;
                    tile_id = head_idx * head_stride + tile_row * head_dim_tiles;
                }
            }
            noc.async_writes_flushed();
            cb_output.pop_front(block_size);
        }
    }
    noc.async_write_barrier();
}
