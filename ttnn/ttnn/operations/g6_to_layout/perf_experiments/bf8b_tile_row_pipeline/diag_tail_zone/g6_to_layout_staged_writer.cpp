// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block writer (BRISC / NoC1): store_block of R1 staged_reorder, packer_untilize and
// sharded_in_place.
//
// bf8b_tile_row_pipeline candidate "rowpipe_tail":
//  - A tiled stage is consumed TILE-ROW by tile-row: the writer waits tile-row i's untilized pages
//    (cumulative cb_wait_front on the block's pages; the block is still flushed and popped once) and
//    its own cb_nan_marks page (one page per live tile-row, see g6_to_layout_bfp8_nan_marks.hpp), then
//    writes its rows while later tile-rows are still being untilized and scanned.
//  - The patch runs only when the page's summary word says the tile-row holds a NaN datum: a clean
//    tile-row costs one wait, one fence, one load and one pop (no record walk, no second fence).
//  - The core's LAST tile-row is written before its marks are checked, so the check overlaps the
//    flight of the core's final writes. If it does hold a NaN: write barrier (the unpatched rows land
//    first), patch, fence, re-write only its live rows that hold one.
// Rows are addressed through get_noc_addr(page, offset): the RM page stride is the aligned page size.
// Zones (profiler builds only), as the baseline: per block wait_row_block (tile-row 0's pages),
// patch_block_nans (tile-row 0's marks), store_block (the rest); the flush is a sum zone.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_bfp8_nan_marks.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool tiled_stage = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 2) != 0;
    constexpr bool patch_bfp8_nans = get_compile_time_arg_val(Geometry::next_ct_arg + 3) != 0;
    constexpr uint32_t cb_nan_marks = get_compile_time_arg_val(Geometry::next_ct_arg + 4);
    constexpr auto output_args = TensorAccessorArgs<Geometry::next_ct_arg + 5>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(output_args, output_addr);

    DeviceZoneScopedN("g6_staged_writer");
    CoreBlocks<sharded> blocks;
    Block block;
    bool more = blocks.next(block);
    while (more) {
        const Block cur = block;
        more = blocks.next(block);
        const bool last_block = !more;
        const uint32_t block_pages = tiled_stage ? moved_tile_rows<sharded>(cur) * Geometry::block_cols_tiles : 1;
        const uint32_t segment_offset = cur.col * TILE_ROW_BYTES;
        const uint32_t segment_bytes = row_segment_bytes(cur);
        auto stage_of = [&](uint32_t i) { return get_read_ptr(cb_row_block) + i * Geometry::stage_tile_row_bytes; };

        auto write_rows = [&](uint32_t i) {
            const uint32_t g = cur.tile_row + i;
            const uint32_t live_rows = live_rows_of(g);
            const uint32_t first_page = first_row_page_of(g);
            uint32_t src = stage_of(i);
            for (uint32_t r = 0; r < live_rows; ++r) {
                noc_async_write(src, output.get_noc_addr(first_page + r, segment_offset), segment_bytes);
                src += Geometry::stage_row_bytes;
            }
        };
        // Tile-row i's marks page. written: its rows are already on the wire (unpatched).
        auto check_marks = [&](uint32_t i, bool written) {
            cb_wait_front(cb_nan_marks, 1);
            // The marks were stored by the reader's RISC; this one may hold stale lines of the page.
            invalidate_l1_cache();
            const uint32_t page = get_read_ptr(cb_nan_marks);
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page) != 0) {
                if (written) {
                    noc_async_write_barrier();
                }
                uint32_t rows = patch_tile_row_nans_rows(page, cur.cols, stage_of(i));
                // Drain the patch stores before the NoC reads the stage.
                invalidate_l1_cache();
                if (written) {
                    // Only live rows: a padding row's page belongs to another row.
                    const uint32_t g = cur.tile_row + i;
                    const uint32_t live_rows = live_rows_of(g);
                    if (live_rows < TILE_HW) {
                        rows &= (1u << live_rows) - 1;
                    }
                    const uint32_t first_page = first_row_page_of(g);
                    while (rows != 0) {
                        const uint32_t r = __builtin_ctz(rows);
                        rows &= rows - 1;
                        noc_async_write(
                            stage_of(i) + r * Geometry::stage_row_bytes,
                            output.get_noc_addr(first_page + r, segment_offset),
                            segment_bytes);
                    }
                }
            }
            cb_pop_front(cb_nan_marks, 1);
        };
        // One tile-row: its pages, then (bf8b) marks before the writes, except the core's last
        // tile-row, whose marks are checked after its writes.
        auto tile_row = [&](uint32_t i) {
            if constexpr (tiled_stage) {
                cb_wait_front(cb_row_block, (i + 1) * Geometry::block_cols_tiles);
            }
            const bool tail = last_block && i + 1 == cur.rows;
            if constexpr (patch_bfp8_nans) {
                if (!tail) {
                    if (i == 0) {
                        DeviceZoneScopedN("patch_block_nans");
                        check_marks(i, false);
                    } else {
                        check_marks(i, false);
                    }
                }
            }
            write_rows(i);
            if constexpr (patch_bfp8_nans) {
                if (tail) {
                    check_marks(i, true);
                }
            }
        };

        {
            DeviceZoneScopedN("wait_row_block");
            cb_wait_front(cb_row_block, tiled_stage ? Geometry::block_cols_tiles : 1);
        }
        {
            DeviceZoneScopedN("store_block");
            tile_row(0);
            for (uint32_t i = 1; i < cur.rows; ++i) {
                tile_row(i);
            }
            {
                DeviceZoneScopedSumN1("row_block_writes_flushed");
                noc_async_writes_flushed();
            }
        }
        if constexpr (tiled_stage) {
            // A ragged interleaved block moves its nominal tile-rows: the stale ones must have landed
            // before the whole block is popped.
            cb_wait_front(cb_row_block, block_pages);
        }
        cb_pop_front(cb_row_block, block_pages);
    }
    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
