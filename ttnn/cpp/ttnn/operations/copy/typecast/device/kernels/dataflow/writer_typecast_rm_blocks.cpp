// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes untilized blocks (32 rows x (block_width_tiles * 32) elements) to row-major sticks.
// Only rows within the logical height and columns within the logical width are written.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_block = get_arg(args::start_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t block_width_tiles = get_arg(args::block_width_tiles);
    constexpr uint32_t element_size = get_arg(args::element_size);
    constexpr uint32_t width = get_arg(args::width);
    constexpr uint32_t height = get_arg(args::height);
    constexpr uint32_t tile_rows_per_batch = get_arg(args::tile_rows_per_batch);
    constexpr uint32_t blocks_per_tile_row = get_arg(args::blocks_per_tile_row);
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t block_row_bytes = block_width_tiles * tile_width * element_size;
    constexpr uint32_t row_bytes = width * element_size;

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const auto s = TensorAccessor(tensor::dst);

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        const uint32_t tile_row = block / blocks_per_tile_row;
        const uint32_t offset_bytes = (block % blocks_per_tile_row) * block_row_bytes;
        const uint32_t valid_bytes =
            row_bytes - offset_bytes < block_row_bytes ? row_bytes - offset_bytes : block_row_bytes;
        const uint32_t first_page = (tile_row / tile_rows_per_batch) * height;
        const uint32_t first_h = (tile_row % tile_rows_per_batch) * tile_height;
        const uint32_t num_rows = height - first_h < tile_height ? height - first_h : tile_height;

        dfb.wait_front(block_width_tiles);
        const uint32_t base = dfb.get_read_ptr();
        for (uint32_t i = 0; i < num_rows; ++i) {
            noc.async_write(
                CoreLocalMem<uint32_t>(base + i * block_row_bytes),
                s,
                valid_bytes,
                {},
                {.page_id = first_page + first_h + i, .offset_bytes = offset_bytes});
        }
        noc.async_write_barrier();
        dfb.pop_front(block_width_tiles);
    }
}
