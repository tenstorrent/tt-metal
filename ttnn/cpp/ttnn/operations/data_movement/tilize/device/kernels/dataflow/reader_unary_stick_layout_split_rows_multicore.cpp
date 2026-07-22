// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "experimental/kernel_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    const auto num_rows = get_arg(args::num_rows);
    const auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    const auto block_width_size = get_arg(args::block_width_size);
    const auto num_full_blocks_in_row = get_arg(args::num_full_blocks_in_row);
    const auto start_page_id = get_arg(args::start_page_id);

    constexpr auto num_pages_in_row =
        get_arg(args::num_pages_in_row);  // For ND-sharded tensors, each row can have multiple pages.
    constexpr auto size_of_valid_data_in_last_page_in_row =
        get_arg(args::size_of_valid_data_in_last_page_in_row);  // For uneven sharding along the width, the last page
                                                                // could contain padding data, so we need to specify the
                                                                // size of valid data we want to read in.

    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);

    auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
        dfb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            // Need an inner loop for pages within row. Only relevant for ND-sharded case on multicore
            // (otherwise this loop only has 1 iteration).
            for (uint32_t l = 0; l < num_pages_in_row; l++) {
                uint32_t width_size =
                    (l == num_pages_in_row - 1) ? size_of_valid_data_in_last_page_in_row : block_width_size;
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(s, dst, width_size, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
                page_id++;
                l1_write_addr += width_size;
            }
        }
        noc.async_read_barrier();
        dfb_in0.push_back(num_tiles);
    };

    uint32_t page_id = start_page_id;
    for (uint32_t i = 0; i < num_rows / tile_height; i++) {
        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, page_id);
        }
        page_id += tile_height * num_pages_in_row;
    }
}
