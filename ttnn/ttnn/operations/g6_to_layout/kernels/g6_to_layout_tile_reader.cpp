// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tile reader (NCRISC / NoC0) of R3 packer_untilize and R4 sharded_in_place: load_block.
//
// R3 (interleaved): per block, one reserve of the nominal block's tile pages in cb_tile_window,
// one whole-tile read per live tile at its block slot (tile-row i, column j -> page
// i * block_cols_tiles + j, the order the untilize consumes), one read barrier, one push. The
// unpacker consumes whole tiles, so a bfp8 tile (shared exponents + datums) is read whole.
// R4 (sharded): cb_tile_window is backed on the resident shard, so the block is already in place;
// the reader publishes the shard's live tile-rows at once. No NoC traffic.
// Both: once a block's tiles are in place, the reader marks its NaN datums on cb_nan_marks (one
// page per block, see g6_to_layout_bfp8_nan_marks.hpp). The scan reads the tiles after they are
// pushed: the consumer only reads them, and their slots are not refilled before the next reserve.
// Zones (profiler builds only): per R3 block, the reserve wait and the block's reads each have a
// zone; the read barrier inside is a sum zone (TT_METAL_PROFILER_SUM). Every block's NaN scan has
// a zone.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_bfp8_nan_marks.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr uint32_t cb_nan_marks = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 3>();

    DeviceZoneScopedN("g6_tile_reader");
    const uint32_t tile_bytes = get_tile_size(cb_tile_window);
    CoreBlocks<sharded> blocks;
    Block block;

    auto mark_nans = [&](uint32_t tiles) {
        DeviceZoneScopedN("mark_block_nans");
        cb_reserve_back(cb_nan_marks, 1);
        mark_block_nans(tiles, block, tile_bytes, get_write_ptr(cb_nan_marks));
        cb_push_back(cb_nan_marks, 1);
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
