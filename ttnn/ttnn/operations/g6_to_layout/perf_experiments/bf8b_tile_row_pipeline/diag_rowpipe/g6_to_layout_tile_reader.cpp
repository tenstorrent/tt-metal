// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tile reader (NCRISC / NoC0) of R3 packer_untilize and R4 sharded_in_place: load_block.
// bf8b_tile_row_pipeline candidate: NaN marks are published per TILE-ROW (one cb_nan_marks page per
// live tile-row of a block, pushed as soon as that tile-row is scanned), so the writer can start on
// tile-row 0 while tile-rows 1.. are still being scanned. The scan's fast path is one OR-reduce over
// the whole tile-row's exponent words (G6_CHEAP_SCAN: 4 ops per word instead of 5).
//
// R3 (interleaved): per block, one reserve of the nominal block's tile pages in cb_tile_window,
// one whole-tile read per live tile at its block slot, one read barrier, one push, then the marks.
// R4 (sharded): cb_tile_window is backed on the resident shard; the reader publishes the shard's
// live tile-rows at once (no NoC traffic), then marks tile-row by tile-row.
// Zones: one mark_block_nans zone per block (around all of its tile-rows), as the baseline.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_bfp8_nan_marks.hpp"

using namespace g6_to_layout;

#ifndef G6_CHEAP_SCAN
#define G6_CHEAP_SCAN 1
#endif

void kernel_main() {
    constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr uint32_t cb_nan_marks = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 3>();

    DeviceZoneScopedN("g6_tile_reader");
    const uint32_t tile_bytes = get_tile_size(cb_tile_window);
    CoreBlocks<sharded> blocks;
    Block block;

    // One cb_nan_marks page per live tile-row of `block`, whose tiles sit at their slots from `tiles`.
    auto mark_nans = [&](uint32_t tiles) {
        DeviceZoneScopedN("mark_block_nans");
        uint32_t row_tiles = tiles;
        for (uint32_t i = 0; i < block.rows; ++i) {
            cb_reserve_back(cb_nan_marks, 1);
            mark_tile_row_nans<G6_CHEAP_SCAN != 0>(row_tiles, block.cols, tile_bytes, get_write_ptr(cb_nan_marks));
            cb_push_back(cb_nan_marks, 1);
            row_tiles += Geometry::block_cols_tiles * tile_bytes;
        }
    };

    if constexpr (sharded) {
        uint32_t live_tiles = 0;
        while (blocks.next(block)) {
            live_tiles += moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles;
        }
        if (live_tiles == 0) {
            return;
        }
        uint32_t tiles = get_write_ptr(cb_tile_window);
        cb_push_back(cb_tile_window, live_tiles);
        // The shard was written by the host, not through this RISC's cache.
        invalidate_l1_cache();
        CoreBlocks<sharded> mark_blocks;
        while (mark_blocks.next(block)) {
            mark_nans(tiles);
            tiles += moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles * tile_bytes;
        }
        return;
    }

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);
    while (blocks.next(block)) {
        const uint32_t block_pages = moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles;
        {
            DeviceZoneScopedN("reserve_tile_window");
            cb_reserve_back(cb_tile_window, block_pages);
        }
        const uint32_t tiles = get_write_ptr(cb_tile_window);
        {
            DeviceZoneScopedN("load_block");
            uint32_t row_slot = tiles;
            for (uint32_t i = 0; i < block.rows; ++i) {
                uint32_t page = (block.tile_row + i) * Geometry::tensor_cols_tiles + block.col;
                uint32_t slot = row_slot;
                for (uint32_t j = 0; j < block.cols; ++j) {
                    noc_async_read(input.get_noc_addr(page), slot, tile_bytes);
                    ++page;
                    slot += tile_bytes;
                }
                row_slot += Geometry::block_cols_tiles * tile_bytes;
            }
            {
                DeviceZoneScopedSumN1("tile_window_read_barrier");
                noc_async_read_barrier();
            }
        }
        cb_push_back(cb_tile_window, block_pages);
        mark_nans(tiles);
    }
}
