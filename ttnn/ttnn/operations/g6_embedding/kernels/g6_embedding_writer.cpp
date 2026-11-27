// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// store_block (op_design.md -> Block schedule): whole-tile writes of one block to the output pages
// g * tensor_col_tiles + c * block_col_tiles + j. Raw NoC writes: kernel_lib has no tile-page writer
// helper (write_sticks_after_untilize writes row-major sticks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tensor_col_tiles = get_common_arg_val<uint32_t>(1);
    const uint32_t tensor_tile_rows = get_common_arg_val<uint32_t>(2);
    const uint32_t col_blocks_per_row = get_common_arg_val<uint32_t>(3);

    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    uint32_t block_row = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - block_row * col_blocks_per_row;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_tile_row = block_row * block_tile_rows;
        const uint32_t left = tensor_tile_rows - first_tile_row;
        const uint32_t tile_rows = left < block_tile_rows ? left : block_tile_rows;
        const uint32_t block_pages = tile_rows * block_col_tiles;

        cb_wait_front(cb_output_tiles, block_pages);
        uint32_t src = get_read_ptr(cb_output_tiles);
        for (uint32_t t = 0; t < tile_rows; ++t) {
            const uint32_t first_page = (first_tile_row + t) * tensor_col_tiles + col_block * block_col_tiles;
            for (uint32_t j = 0; j < block_col_tiles; ++j) {
                noc_async_write(src, output_accessor.get_noc_addr(first_page + j), tile_bytes);
                src += tile_bytes;
            }
        }
        // The slots go back to compute only once the NoC has read them out of L1.
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, block_pages);

        if (col_block + 1 == col_blocks_per_row) {
            col_block = 0;
            ++block_row;
        } else {
            ++col_block;
        }
    }
    noc_async_write_barrier();
}
