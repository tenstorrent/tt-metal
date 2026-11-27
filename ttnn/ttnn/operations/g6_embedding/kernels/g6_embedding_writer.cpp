// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer (BRISC, NoC1): store_block — whole tile pages of a block.
//
// Raw NoC writes instead of dataflow_kernel_lib::write_sticks_after_untilize: that helper writes
// row-major sticks; the result here is tile pages, and kernel_lib has no tile-page writer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    constexpr uint32_t col_blocks_per_row = get_arg(args::col_blocks_per_row);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);
    constexpr uint32_t tensor_tile_rows = get_arg(args::tensor_tile_rows);
    constexpr uint32_t tensor_col_tiles = get_arg(args::tensor_col_tiles);
    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    Noc noc;
    DataflowBuffer output_tiles(dfb::output_tiles);
    const auto output_accessor = TensorAccessor(tensor::output);
    const uint32_t tile_bytes = output_tiles.get_tile_size();

    uint32_t group = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - group * col_blocks_per_row;

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        DeviceZoneScopedN("store_block");
        const uint32_t first_row = group * block_tile_rows;
        const uint32_t remaining = tensor_tile_rows - first_row;
        const uint32_t rows = remaining < block_tile_rows ? remaining : block_tile_rows;
        const uint32_t pages = rows * block_col_tiles;

        output_tiles.wait_front(pages);
        for (uint32_t j = 0; j < rows; ++j) {
            // Output tile (g, k) is page g * tensor_col_tiles + k, g counting per-image tile-rows.
            const uint32_t first_page = (first_row + j) * tensor_col_tiles + col_block * block_col_tiles;
            for (uint32_t t = 0; t < block_col_tiles; ++t) {
                noc.async_write(
                    output_tiles,
                    output_accessor,
                    tile_bytes,
                    {.offset_bytes = (j * block_col_tiles + t) * tile_bytes},
                    {.page_id = first_page + t});
            }
        }
        // The slots are handed back to compute only once the NoC has read them out of L1.
        noc.async_writes_flushed();
        output_tiles.pop_front(pages);

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++group;
        }
    }
    noc.async_write_barrier();
}
