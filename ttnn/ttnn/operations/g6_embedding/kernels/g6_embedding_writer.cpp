// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer: store_block. Raw NoC writes: kernel_lib has no tile-page writer helper
// (write_sticks_after_untilize writes row-major sticks). One whole-tile write per output page,
// one flush per block before the pop (the slot must leave L1 before compute reuses it), one
// write barrier before return so no transaction outlives the program.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t col_blocks_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();

    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);

    uint32_t row_block = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_block * col_blocks_per_row;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        cb_wait_front(cb_output_tiles, block_pages);
        uint32_t src = get_read_ptr(cb_output_tiles);
        const uint32_t first_tile_row = row_block * block_tile_rows;
        for (uint32_t t = 0; t < block_tile_rows; ++t) {
            const uint32_t tile_row = first_tile_row + t;
            if (tile_row >= tensor_tile_rows) {
                break;
            }
            const uint32_t first_page = tile_row * tensor_col_tiles + col_block * block_col_tiles;
            for (uint32_t j = 0; j < block_col_tiles; ++j) {
                noc_async_write(src, output_accessor.get_noc_addr(first_page + j), tile_bytes);
                src += tile_bytes;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, block_pages);

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++row_block;
        }
    }
    noc_async_write_barrier();
}
