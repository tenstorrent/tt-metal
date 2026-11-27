// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer: store_block. Whole 2 KB tiles, one write per tile; one flush per block
// before the slots are released, one write barrier before return.
//
// No tile-page writer helper exists in kernel_lib (write_sticks_after_untilize writes row-major
// sticks), so the writes are raw NoC calls.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t TILE_BYTES = 32 * 32 * 2;
}  // namespace

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);

    const uint32_t col_blocks_per_row = get_arg(args::col_blocks_per_row);
    const uint32_t tensor_tile_rows = get_arg(args::tensor_tile_rows);
    const uint32_t tensor_col_tiles = get_arg(args::tensor_col_tiles);
    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    Noc noc;
    DataflowBuffer output_tiles(dfb::output_tiles);
    const auto output_accessor = TensorAccessor(tensor::output);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block = first_block + block_idx;
        const uint32_t row_block = block / col_blocks_per_row;
        const uint32_t col_block = block - row_block * col_blocks_per_row;
        const uint32_t remaining_tile_rows = tensor_tile_rows - row_block * block_tile_rows;
        const uint32_t num_tile_rows = remaining_tile_rows < block_tile_rows ? remaining_tile_rows : block_tile_rows;
        const uint32_t block_pages = num_tile_rows * block_col_tiles;

        output_tiles.wait_front(block_pages);
        for (uint32_t t = 0; t < num_tile_rows; ++t) {
            const uint32_t tile_row = row_block * block_tile_rows + t;
            const uint32_t first_page = tile_row * tensor_col_tiles + col_block * block_col_tiles;
            for (uint32_t j = 0; j < block_col_tiles; ++j) {
                noc.async_write<NocOptions::DEFAULT, TILE_BYTES>(
                    output_tiles,
                    output_accessor,
                    TILE_BYTES,
                    {.offset_bytes = (t * block_col_tiles + j) * TILE_BYTES},
                    {.page_id = first_page + j});
            }
        }
        // The slots are reused by compute once popped: the writes must have left L1 first.
        noc.async_writes_flushed();
        output_tiles.pop_front(block_pages);
    }
    noc.async_write_barrier();
}
