// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tile reader (NCRISC / NoC0) of the untilize regimes: load_block.
//
// R1 (bf16) and R3 (bfp8), interleaved: per block, one reserve of the nominal block's tile pages in
// cb_tile_window, the block's live tiles read to their block slot (tile-row i, column j -> page
// i * block_cols_tiles + j, the order the untilize consumes), one read barrier, one push.
//   bf16: only each tile's live face-row prefixes are read (adjacent whole faces merged, so an
//   interior tile is one 2048-B read). The slot's other bytes are stale and untilize into stage bytes
//   the writer never writes (rows >= live_rows, bytes past 2 * W).
//   bfp8: the unpacker consumes whole tiles (shared exponents + datums), so a tile is read whole.
// R4 (sharded): cb_tile_window is backed on the resident shard, so the block is already in place;
// the reader publishes the shard's live tile-rows at once. No NoC traffic.
// bfp8 only: once a block's tiles are in place, the reader marks its NaN datums on cb_nan_marks (one
// page per block, see g6_to_layout_bfp8_nan_marks.hpp). The scan reads the tiles after they are
// pushed: the consumer only reads them, and their slots are not refilled before the next reserve.
// Zones (profiler builds only): per interleaved block, the reserve wait and the block's reads each
// have a zone; the read barrier inside is a sum zone (TT_METAL_PROFILER_SUM). Every block's NaN scan
// has a zone.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_bfp8_nan_marks.hpp"

using namespace g6_to_layout;

namespace {

// Live face-row prefixes of one bf16 tile into a 2048-B slot at the same face offsets; adjacent whole
// faces are merged into one read.
FORCE_INLINE void read_live_tile(uint64_t src, uint32_t dst, uint32_t live_rows, uint32_t live_cols) {
    const uint32_t top_rows = min_u32(live_rows, FACE_HW);
    const uint32_t bottom_rows = live_rows - top_rows;
    if (live_cols > FACE_HW) {
        if (bottom_rows == FACE_HW) {
            noc_async_read(src, dst, TILE_BYTES);
        } else if (bottom_rows > 0) {
            noc_async_read(src, dst, 2 * FACE_BYTES + bottom_rows * FACE_ROW_BYTES);
            noc_async_read(src + 3 * FACE_BYTES, dst + 3 * FACE_BYTES, bottom_rows * FACE_ROW_BYTES);
        } else if (top_rows == FACE_HW) {
            noc_async_read(src, dst, 2 * FACE_BYTES);
        } else {
            noc_async_read(src, dst, top_rows * FACE_ROW_BYTES);
            noc_async_read(src + FACE_BYTES, dst + FACE_BYTES, top_rows * FACE_ROW_BYTES);
        }
    } else {
        noc_async_read(src, dst, top_rows * FACE_ROW_BYTES);
        if (bottom_rows > 0) {
            noc_async_read(src + 2 * FACE_BYTES, dst + 2 * FACE_BYTES, bottom_rows * FACE_ROW_BYTES);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr bool bfp8 = get_compile_time_arg_val(Geometry::next_ct_arg + 2) != 0;
    constexpr uint32_t cb_nan_marks = get_compile_time_arg_val(Geometry::next_ct_arg + 3);
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 4>();

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
                const uint32_t live_rows = live_rows_of(block.tile_row + i);
                for (uint32_t j = 0; j < block.cols; ++j) {
                    if constexpr (bfp8) {
                        noc_async_read(input.get_noc_addr(page), slot, tile_bytes);
                    } else {
                        read_live_tile(input.get_noc_addr(page), slot, live_rows, live_cols_of(block.col + j));
                    }
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
        if constexpr (bfp8) {
            mark_nans(tiles);
        }
    }
}
