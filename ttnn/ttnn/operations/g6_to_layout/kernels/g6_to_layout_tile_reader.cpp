// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tile reader (NCRISC / NoC0) of R3 packer_untilize and R4 sharded_in_place: load_block.
//
// R3 (interleaved): per block, one reserve of the nominal block's tile pages in cb_tile_window,
// one whole-tile read per live tile at its block slot (tile-row i, column j -> page
// i * block_cols_tiles + j, the order the untilize consumes), one read barrier, one push. The
// unpacker consumes whole tiles, so a bfp8 tile (shared exponents + datums) is read whole.
// R4 (sharded): cb_tile_window is backed on the resident shard, so the block is already in place;
// the reader only publishes the shard's live tile-rows. No NoC traffic.
// Zones (profiler builds only): per R3 block, the reserve wait and the block's reads each have a
// zone; the read barrier inside is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 2>();

    DeviceZoneScopedN("g6_tile_reader");
    CoreBlocks<sharded> blocks;
    Block block;

    if constexpr (sharded) {
        uint32_t live_tiles = 0;
        while (blocks.next(block)) {
            live_tiles += moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles;
        }
        if (live_tiles > 0) {
            cb_push_back(cb_tile_window, live_tiles);
        }
        return;
    }

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(cb_tile_window);
    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);
    while (blocks.next(block)) {
        const uint32_t block_pages = moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles;
        {
            DeviceZoneScopedN("reserve_tile_window");
            cb_reserve_back(cb_tile_window, block_pages);
        }
        {
            DeviceZoneScopedN("load_block");
            uint32_t row_slot = get_write_ptr(cb_tile_window);
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
    }
}
