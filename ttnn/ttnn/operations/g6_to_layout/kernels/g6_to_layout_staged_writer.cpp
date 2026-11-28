// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block writer (BRISC / NoC1) of the untilize regimes (R1 staged_reorder, R3 packer_untilize,
// R4 sharded_in_place): store_block.
//
// The row stage is consumed tile-row by tile-row: the writer waits tile-row i's untilized pages
// (block_cols_tiles of them, 32 rows of block_cols_tiles * 64 B; a cumulative cb_wait_front on the
// block) and issues one write per live row carrying the block's live bytes of that row. The block is
// flushed and popped once. The segment ends at the row's last live byte, so stale stage bytes (padding
// columns, unread faces, untilized stale tiles) never leave L1.
// A bfloat8_b input (patch_bfp8_nans): the NaN marks never gate a write. Tile-row i's cb_nan_marks
// page is checked right after its rows are issued; a clean tile-row (summary word 0, the common case)
// costs one wait, one fence, one load and one pop. A tile-row holding a NaN datum: write barrier (its
// unpatched rows land first), the host's NaN rendering stored over the marked datums in the stage,
// fence, and only its live rows that hold one written again (g6_to_layout_bfp8_nan_marks.hpp). The
// stage is popped after the block's writes are flushed, so it is intact for any rewrite.
// Rows are issued per tile-row by row_issue::write_rows (g6_to_layout_row_issue.hpp): consecutive
// pages, the RM page stride being the aligned page size, not 2 * W.
// Zones (profiler builds only): per block, the wait and the block's work each have a zone; the
// flush inside the work is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_bfp8_nan_marks.hpp"
#include "g6_to_layout_row_issue.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr bool sharded = get_compile_time_arg_val(Geometry::next_ct_arg + 1) != 0;
    constexpr bool patch_bfp8_nans = get_compile_time_arg_val(Geometry::next_ct_arg + 2) != 0;
    constexpr uint32_t cb_nan_marks = get_compile_time_arg_val(Geometry::next_ct_arg + 3);
    constexpr auto output_args = TensorAccessorArgs<Geometry::next_ct_arg + 4>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(output_args, output_addr);

    DeviceZoneScopedN("g6_staged_writer");
    CoreBlocks<sharded> blocks;
    Block block;
    while (blocks.next(block)) {
        const uint32_t block_pages = moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles;
        const uint32_t segment_offset = block.col * TILE_ROW_BYTES;
        const uint32_t segment_bytes = row_segment_bytes(block);
        const uint32_t stage = get_read_ptr(cb_row_block);

        // Tile-row i's rows are already on the wire, unpatched.
        auto check_marks = [&](uint32_t i) __attribute__((always_inline)) {
            cb_wait_front(cb_nan_marks, 1);
            // The marks were stored by the reader's RISC; this one may hold stale lines of the page.
            invalidate_l1_cache();
            const uint32_t page = get_read_ptr(cb_nan_marks);
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page) != 0) {
                noc_async_write_barrier();
                const uint32_t tile_row_stage = stage + i * Geometry::stage_tile_row_bytes;
                uint32_t rows = patch_tile_row_nans_rows(page, block.cols, tile_row_stage);
                // Drain the patch stores before the NoC reads the stage.
                invalidate_l1_cache();
                const uint32_t g = block.tile_row + i;
                const uint32_t live_rows = live_rows_of(g);
                if (live_rows < TILE_HW) {
                    // A padding row's page belongs to another row (or to no row).
                    rows &= (1u << live_rows) - 1;
                }
                const uint32_t first_page = first_row_page_of(g);
                while (rows != 0) {
                    const uint32_t r = __builtin_ctz(rows);
                    rows &= rows - 1;
                    noc_async_write(
                        tile_row_stage + r * Geometry::stage_row_bytes,
                        output.get_noc_addr(first_page + r, segment_offset),
                        segment_bytes);
                }
            }
            cb_pop_front(cb_nan_marks, 1);
        };

        {
            DeviceZoneScopedN("wait_row_block");
            cb_wait_front(cb_row_block, Geometry::block_cols_tiles);
        }
        {
            DeviceZoneScopedN("store_block");
            for (uint32_t i = 0; i < block.rows; ++i) {
                if (i > 0) {
                    cb_wait_front(cb_row_block, (i + 1) * Geometry::block_cols_tiles);
                }
                const uint32_t g = block.tile_row + i;
                row_issue::write_rows(
                    output,
                    first_row_page_of(g),
                    live_rows_of(g),
                    segment_offset,
                    segment_bytes,
                    stage + i * Geometry::stage_tile_row_bytes,
                    Geometry::stage_row_bytes);
                if constexpr (patch_bfp8_nans) {
                    check_marks(i);
                }
            }
            {
                DeviceZoneScopedSumN1("row_block_writes_flushed");
                noc_async_writes_flushed();
            }
        }
        // An interleaved block moves its nominal tile-rows: the stale ones must have landed before the
        // whole block is popped.
        cb_wait_front(cb_row_block, block_pages);
        cb_pop_front(cb_row_block, block_pages);
    }
    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
