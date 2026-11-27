// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer (BRISC / NoC1): store_block.
//
// Writes each block's tiles as whole 2 KB pages: tile j of tile-row g in column block c is output
// page g * tensor_col_tiles + c * block_col_tiles + j. No tile-page writer helper exists in
// kernel_lib (write_sticks_after_untilize writes row-major sticks).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tensor_col_tiles = get_common_arg_val<uint32_t>(1);
    const uint32_t tensor_tile_rows = get_common_arg_val<uint32_t>(2);
    const uint32_t col_blocks_per_row = get_common_arg_val<uint32_t>(3);

    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);

    uint32_t row_group = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_group * col_blocks_per_row;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        {
            DeviceZoneScopedN("store_block_wait");
            cb_wait_front(cb_output_tiles, block_pages);
        }
        {
            DeviceZoneScopedN("store_block");
            uint32_t l1_tile = get_read_ptr(cb_output_tiles);
            for (uint32_t t = 0; t < block_tile_rows; ++t) {
                const uint32_t g = row_group * block_tile_rows + t;
                if (g >= tensor_tile_rows) {
                    break;
                }
                uint32_t page = g * tensor_col_tiles + col_block * block_col_tiles;
                for (uint32_t j = 0; j < block_col_tiles; ++j) {
                    noc_async_write(l1_tile, output_accessor.get_noc_addr(page), tile_bytes);
                    l1_tile += tile_bytes;
                    ++page;
                }
            }
            // The slots are handed back to compute only after their bytes have left L1 (BB-06).
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_tiles, block_pages);

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++row_group;
        }
    }
    noc_async_write_barrier();
}
