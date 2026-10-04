// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

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

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer cb_in0(dfb::in);

#ifdef ARCH_QUASAR
    // TEMP DIAGNOSTIC (cross-test hang): the reader's launch-populated base for dfb::in. Gated to SMALL
    // inputs only (<=32 tiles/block) so test_concat.py's 16 wide (256-tile) builds don't flood the print
    // buffer and drown out test_concat_small_grid's hanging tilize. See
    // project_quasar_graphcase_tile_reroute. "DONE" marks the reader finished all async reads+barriers.
    if (num_tiles_per_block <= 32) {
        DPRINT(
            "QSR tilize reader DFB base: in={} num_rows={} ntpb={} nfbir={}\n",
            get_local_dfb_interface(static_cast<uint32_t>(dfb::in)).tc_slots[0].base_addr,
            num_rows,
            num_tiles_per_block,
            num_full_blocks_in_row);
    }
#endif

    auto read_tiles = [&](const uint32_t& num_tiles, uint32_t page_id) {
        cb_in0.reserve_back(num_tiles);
        uint32_t dst_offset = 0;
        for (uint32_t k = 0; k < tile_height; k++) {
            // Need an inner loop for pages within row. Only relevant for ND-sharded case on multicore
            // (otherwise this loop only has 1 iteration).
            for (uint32_t l = 0; l < num_pages_in_row; l++) {
                uint32_t width_size =
                    (l == num_pages_in_row - 1) ? size_of_valid_data_in_last_page_in_row : block_width_size;
                noc.async_read(
                    s, cb_in0, width_size, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = dst_offset});
                page_id++;
                dst_offset += width_size;
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

    // Drain-on-exit: wait until the consuming UNPACK has acked every pushed tile so this input DFB's
    // tile counter leaves balanced (posted==acked), rather than relying on natural pipeline drain. The
    // quasar tilize kernels historically omit this (unlike binary_ng); a predecessor that exits non-idle
    // leaves the next program's unpack reading a stale counter (see tile_counter_reset_issue.md).
#ifdef ARCH_QUASAR
    cb_in0.finish();
    if (num_tiles_per_block <= 32) {
        DPRINT("QSR tilize reader: DONE (all reads+push+finish)\n");
    }
#endif
}
