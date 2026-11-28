// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block writer (BRISC / NoC1): store_block of R1 staged_reorder, packer_untilize and
// sharded_in_place.
// bf8b_tile_row_pipeline candidate: a tiled stage is consumed TILE-ROW by tile-row: the writer waits
// tile-row i's untilized pages (cumulative cb_wait_front) and its own cb_nan_marks page, patches only
// when the page's summary word says the tile-row holds a NaN (else no patch walk and no second
// fence), and writes its rows while later tile-rows are still being untilized / scanned. The block is
// still flushed and popped once. Zone names match the baseline (wait_row_block = tile-row 0's pages,
// patch_block_nans = tile-row 0's marks + patch, store_block = all writes + later tile-rows' waits).
//
// Per block: one cb_wait_front, one write per live row carrying the block's live bytes of that
// row, one noc_async_writes_flushed, one cb_pop_front. The segment ends at the row's last live
// byte, so stale stage bytes (padding columns, unread faces, untilized stale tiles) never leave L1.
// The block's stage is one whole-block page (R1) or block_cols_tiles tile pages per tile-row as the
// untilize helper packs them (tiled_stage); its layout is the same either way.
// A bfloat8_b input (patch_bfp8_nans) also waits the block's cb_nan_marks page and stores the host's
// NaN rendering over the marked datums before any row is written (g6_to_layout_bfp8_nan_marks.hpp).
// Rows are addressed through get_noc_addr(page, offset): the RM page stride is the aligned
// page size, not 2 * W.
// Zones (profiler builds only): per block, the wait and the block's work each have a zone; the
// flush inside the work is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_bfp8_nan_marks.hpp"

using namespace g6_to_layout;

// G6_DUAL_NOC: alternate the row writes between NoC1 (the writer's) and NoC0 (unused by the reader
// of R4, read-only traffic in R3). BRISC's software counters of NoC0 are synced to the NIU's at
// kernel start (dedicated NoC mode only initializes the kernel's own NoC).
#ifndef G6_DUAL_NOC
#define G6_DUAL_NOC 0
#endif
constexpr bool kDualNoc = G6_DUAL_NOC != 0;
// G6_ROW_ROTATE: each tile-row's rows are written starting at row (first_page / NUM_L1_BANKS) % live_rows
// and wrapping. Cores whose first pages are congruent modulo the bank count would otherwise hit the
// same destination bank at every step for their whole run (c37: shard k and k + 55, 96 * 55 = 48 * 110);
// their page-wrap counts differ, so the rotation de-synchronizes them.
#ifndef G6_ROW_ROTATE
#define G6_ROW_ROTATE 0
#endif
constexpr bool kRowRotate = G6_ROW_ROTATE != 0;
constexpr uint8_t kOtherNoc = 1 - noc_index;

void kernel_main() {
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool tiled_stage = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 2) != 0;
    constexpr bool patch_bfp8_nans = get_compile_time_arg_val(Geometry::next_ct_arg + 3) != 0;
    constexpr uint32_t cb_nan_marks = get_compile_time_arg_val(Geometry::next_ct_arg + 4);
    constexpr auto output_args = TensorAccessorArgs<Geometry::next_ct_arg + 5>();

    // Code-layout probe: shifts every later instruction by 4 bytes (executed once).
    asm volatile("nop");
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(output_args, output_addr);

    DeviceZoneScopedN("g6_staged_writer");
    if constexpr (kDualNoc) {
        noc_local_state_init(kOtherNoc);
    }
    CoreBlocks<sharded> blocks;
    Block block;
    while (blocks.next(block)) {
        const uint32_t block_pages = tiled_stage ? moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles : 1;
        const uint32_t segment_offset = block.col * TILE_ROW_BYTES;
        const uint32_t segment_bytes = row_segment_bytes(block);

        // Tile-row i of the block: wait its untilized pages (cumulative wait on the block's pages, so
        // the block is still popped once, after its writes are flushed) and, for bf8b, its marks page.
        // Returns the tile-row's stage address.
        auto ready_tile_row = [&](uint32_t i) -> uint32_t {
            if constexpr (tiled_stage) {
                cb_wait_front(cb_row_block, (i + 1) * Geometry::block_cols_tiles);
            } else if (i == 0) {
                cb_wait_front(cb_row_block, 1);
            }
            const uint32_t stage = get_read_ptr(cb_row_block) + i * Geometry::stage_tile_row_bytes;
            if constexpr (patch_bfp8_nans) {
                cb_wait_front(cb_nan_marks, 1);
                // The marks were stored by the reader's RISC; this one may hold stale lines of the page.
                invalidate_l1_cache();
                const uint32_t page = get_read_ptr(cb_nan_marks);
                if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page) != 0) {
                    patch_tile_row_nans(page, block.cols, stage);
                    // Drain the patch stores before the NoC reads the stage.
                    invalidate_l1_cache();
                }
                cb_pop_front(cb_nan_marks, 1);
            }
            return stage;
        };
        auto store_tile_row = [&](uint32_t i, uint32_t src) {
            const uint32_t g = block.tile_row + i;
            const uint32_t live_rows = live_rows_of(g);
            const uint32_t first_page = first_row_page_of(g);
            if constexpr (kRowRotate) {
                const uint32_t stage = src;
                uint32_t r = (first_page / NUM_L1_BANKS) % live_rows;
                src = stage + r * Geometry::stage_row_bytes;
                for (uint32_t n = 0; n < live_rows; ++n) {
                    noc_async_write(src, output.get_noc_addr(first_page + r, segment_offset), segment_bytes);
                    if (++r == live_rows) {
                        r = 0;
                        src = stage;
                    } else {
                        src += Geometry::stage_row_bytes;
                    }
                }
            } else {
                for (uint32_t r = 0; r < live_rows; ++r) {
                    const uint8_t noc = (kDualNoc && (r & 1)) ? kOtherNoc : noc_index;
                    noc_async_write(src, output.get_noc_addr(first_page + r, segment_offset, noc), segment_bytes, noc);
                    src += Geometry::stage_row_bytes;
                }
            }
        };

        uint32_t stage0;
        {
            DeviceZoneScopedN("wait_row_block");
            if constexpr (tiled_stage) {
                cb_wait_front(cb_row_block, Geometry::block_cols_tiles);
            } else {
                cb_wait_front(cb_row_block, 1);
            }
        }
        {
            DeviceZoneScopedN("patch_block_nans");
            stage0 = ready_tile_row(0);
        }
        {
            DeviceZoneScopedN("store_block");
            store_tile_row(0, stage0);
            for (uint32_t i = 1; i < block.rows; ++i) {
                store_tile_row(i, ready_tile_row(i));
            }
            {
                DeviceZoneScopedSumN1("row_block_writes_flushed");
                noc_async_writes_flushed();
                if constexpr (kDualNoc) {
                    noc_async_writes_flushed(kOtherNoc);
                }
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
        if constexpr (kDualNoc) {
            noc_async_write_barrier(kOtherNoc);
        }
    }
}
