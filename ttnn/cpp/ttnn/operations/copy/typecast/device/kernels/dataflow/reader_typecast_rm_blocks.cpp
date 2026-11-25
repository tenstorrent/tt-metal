// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads row-major sticks into tilize blocks: 32 rows x (block_width_tiles * 32) elements per block.
// Rows past the logical height and columns past the logical width are zero-filled, so a block-float
// pack of the tilized data never shares an exponent with stale padding.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

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
    DataflowBuffer dfb(dfb::in);
    const auto s = TensorAccessor(tensor::src);

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        const uint32_t tile_row = block / blocks_per_tile_row;
        const uint32_t offset_bytes = (block % blocks_per_tile_row) * block_row_bytes;
        const uint32_t valid_bytes =
            row_bytes - offset_bytes < block_row_bytes ? row_bytes - offset_bytes : block_row_bytes;
        const uint32_t first_page = (tile_row / tile_rows_per_batch) * height;
        const uint32_t first_h = (tile_row % tile_rows_per_batch) * tile_height;

        dfb.reserve_back(block_width_tiles);
        const uint32_t base = dfb.get_write_ptr();
        for (uint32_t i = 0; i < tile_height; ++i) {
            const uint32_t dst = base + i * block_row_bytes;
            if (first_h + i < height) {
                dataflow_kernel_lib::fill_l1_range<element_size>(dst + valid_bytes, block_row_bytes - valid_bytes, 0);
                noc.async_read(
                    s,
                    CoreLocalMem<uint32_t>(dst),
                    valid_bytes,
                    {.page_id = first_page + first_h + i, .offset_bytes = offset_bytes},
                    {});
            } else {
                dataflow_kernel_lib::fill_l1_range<element_size>(dst, block_row_bytes, 0);
            }
        }
        noc.async_read_barrier();
        dfb.push_back(block_width_tiles);
    }
}
