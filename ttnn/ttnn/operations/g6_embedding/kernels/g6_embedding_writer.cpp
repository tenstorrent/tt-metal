// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer (BRISC / NoC1): store_block.
//
// Walks the same contiguous block-id run as the reader. Output tile (g, k) is page
// g * tensor_col_tiles + k, g = image * ceil(S/32) + tr, so each image's partial last tile-row
// keeps its own padding. Whole 2 KB tiles, one write each; a slot is popped once its writes have
// left L1 (flushed), and the kernel retires every write before returning (BB-03).
//
// Raw NoC writes: kernel_lib has no tile-page writer helper (write_sticks_after_untilize writes
// row-major sticks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#if defined(G6_EMBEDDING_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t col_blocks_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();

    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto output_accessor = TensorAccessor(output_args, output_addr);

    auto tile_rows_in = [](uint32_t row_block) -> uint32_t {
        const uint32_t remaining = tensor_tile_rows - row_block * block_tile_rows;
        return remaining < block_tile_rows ? remaining : block_tile_rows;
    };

    uint32_t row_block = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_block * col_blocks_per_row;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        {
            G6_ZONE("store_wait");
            cb_wait_front(cb_output_tiles, block_pages);
        }
        {
            G6_ZONE("store_block");
            uint32_t src = get_read_ptr(cb_output_tiles);
            const uint32_t tile_rows = tile_rows_in(row_block);
            for (uint32_t t = 0; t < tile_rows; ++t) {
                const uint32_t first_page =
                    (row_block * block_tile_rows + t) * tensor_col_tiles + col_block * block_col_tiles;
                for (uint32_t j = 0; j < block_col_tiles; ++j) {
                    noc_async_write(src, output_accessor.get_noc_addr(first_page + j), tile_bytes);
                    src += tile_bytes;
                }
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_tiles, block_pages);

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++row_block;
        }
    }
    noc_async_write_barrier();
}
