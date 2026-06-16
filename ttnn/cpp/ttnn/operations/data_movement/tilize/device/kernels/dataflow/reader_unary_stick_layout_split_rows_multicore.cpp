// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port. Used only by the TilizeMultiCoreDefault factory, so ported in place.
// Logic unchanged from the legacy reader; only the access mechanism moves to named bindings:
// source tensor address -> ta::input, CB id -> dfb::src0, positional compile-time / runtime args ->
// get_arg(args::...). The legacy unused runtime-arg slots 2/6/7 (stick_size, num_leftover_tiles,
// leftover_width_in_row) and compile-time slot 0 (aligned_page_size) are dropped — never read here.

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = dfb::src0;
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t num_tiles_per_block = get_arg(args::num_tiles_per_block);
    const uint32_t block_width_size = get_arg(args::block_width_size);
    const uint32_t num_full_blocks_in_row = get_arg(args::num_full_blocks_in_row);
    const uint32_t start_page_id = get_arg(args::start_page_id);

    constexpr uint32_t num_pages_in_row =
        get_arg(args::num_pages_in_row);  // For ND-sharded tensors, each row can have multiple pages.
    constexpr uint32_t size_of_valid_data_in_last_page_in_row =
        get_arg(args::size_of_valid_data_in_last_page_in_row);  // For uneven sharding along the width, the last page
                                                                // could contain padding data, so we need to specify the
                                                                // size of valid data we want to read in.

    const auto s = TensorAccessor(ta::input);

    Noc noc;
    DataflowBuffer cb_in0(cb_id_in0);

    auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
        cb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
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
        cb_in0.push_back(num_tiles);
    };

    uint32_t page_id = start_page_id;
    for (uint32_t i = 0; i < num_rows / tile_height; i++) {
        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, page_id);
        }
        page_id += tile_height * num_pages_in_row;
    }
}
