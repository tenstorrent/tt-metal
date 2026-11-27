// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer of g6_embedding: store_block. Whole 2 KB tiles, one write per tile.
//
// Raw NoC writes: the kernel library has no tile-page writer helper (write_sticks_after_untilize
// writes row-major sticks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef G6_EMBEDDING_ZONES
#include "api/debug/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);
    constexpr uint32_t col_blocks_per_row = get_arg(args::col_blocks_per_row);
    constexpr uint32_t tensor_tile_rows = get_arg(args::tensor_tile_rows);
    constexpr uint32_t tensor_col_tiles = get_arg(args::tensor_col_tiles);
    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    DataflowBuffer dfb_out(dfb::output_tiles);
    const auto out_acc = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_out.get_entry_size();

    uint32_t row_block = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_block * col_blocks_per_row;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        {
            G6_ZONE("out_wait");
            dfb_out.wait_front(block_pages);
        }
        {
            G6_ZONE("store_block");
            const uint32_t block_base = dfb_out.get_read_ptr();
            for (uint32_t t = 0; t < block_tile_rows; ++t) {
                const uint32_t g = row_block * block_tile_rows + t;
                if (g >= tensor_tile_rows) {
                    break;
                }
                const uint32_t first_page = g * tensor_col_tiles + col_block * block_col_tiles;
                const uint32_t tile_row_base = block_base + t * block_col_tiles * tile_bytes;
                for (uint32_t j = 0; j < block_col_tiles; ++j) {
                    noc_async_write(tile_row_base + j * tile_bytes, out_acc.get_noc_addr(first_page + j), tile_bytes);
                }
            }
            // The slot is handed back to compute only after its bytes have left L1.
            noc_async_writes_flushed();
        }
        dfb_out.pop_front(block_pages);
        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++row_block;
        }
    }
    G6_ZONE("write_barrier");
    noc_async_write_barrier();
}
