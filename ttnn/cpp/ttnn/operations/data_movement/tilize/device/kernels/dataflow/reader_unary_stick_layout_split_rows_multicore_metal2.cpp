// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_stick_layout_split_rows_multicore.cpp.
//
// Multi-core variant: reads row-major sticks from interleaved input across
// multiple ND-sharded pages per row (with last-page valid-size truncation).
//
// Bindings (named, from host KernelSpec):
//   dfb::input                                   — DFB endpoint (PRODUCER)
//   ta::input                                    — TensorAccessor (input)
//   args::num_pages_in_row                       — CTA
//   args::size_of_valid_data_in_last_page_in_row — CTA
//   args::num_rows                               — RTA
//   args::num_tiles_per_block                    — RTA
//   args::block_width_size                       — RTA
//   args::num_full_blocks_in_row                 — RTA
//   args::start_page_id                          — RTA
//
// Dead-RTA-slot note: legacy slots 2 (page_size), 6, 7 unread by kernel — dropped.
// Legacy CTA[0]=aligned_page_size was also unread — dropped.

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    constexpr auto num_pages_in_row = get_arg(args::num_pages_in_row);
    constexpr auto size_of_valid_data_in_last_page_in_row = get_arg(args::size_of_valid_data_in_last_page_in_row);

    auto num_rows = get_arg(args::num_rows);
    auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    auto block_width_size = get_arg(args::block_width_size);
    auto num_full_blocks_in_row = get_arg(args::num_full_blocks_in_row);
    auto start_page_id = get_arg(args::start_page_id);

    const auto s = TensorAccessor(ta::input);

    Noc noc;
    DataflowBuffer cb_in0(dfb::input);

    auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
        cb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
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
