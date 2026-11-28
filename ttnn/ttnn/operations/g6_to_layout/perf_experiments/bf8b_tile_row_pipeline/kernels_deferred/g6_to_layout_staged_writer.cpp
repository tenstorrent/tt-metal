// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block writer (BRISC / NoC1): store_block of R1 staged_reorder, packer_untilize and
// sharded_in_place.
//
// bf8b_tile_row_pipeline candidate "deferred patch": a tiled stage is consumed TILE-ROW by tile-row
// (cumulative cb_wait_front on the block's pages), and each tile-row's rows go out as soon as it is
// untilized, WITHOUT waiting for its NaN marks. The marks of tile-row i are checked after tile-row
// i + 1's writes are issued (the last one after the block's last writes). A clean tile-row (summary
// word 0, the common case) costs one wait, one fence, one load and one pop. A tile-row holding a NaN
// datum: write barrier (its unpatched rows must land before their re-writes), store 0xFFFF over the
// marked datums in the stage, fence, re-write only the rows that hold one. The reader's scan is thus
// off the writer's critical path; the stage is popped once per block after its writes are flushed, so
// it is intact for any re-write.
// Rows are addressed through get_noc_addr(page, offset): the RM page stride is the aligned page size.
// Zones (profiler builds only): per block wait_row_block (tile-row 0's pages), store_block (the rest),
// patch_block_nans (around the marks check of tile-row 0, as many zones as the baseline), and
// repatch_nan_rows only when a tile-row holds a NaN.

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
    while (blocks.next(block)) {
        const uint32_t block_pages = tiled_stage ? moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles : 1;
        const uint32_t segment_offset = block.col * TILE_ROW_BYTES;
        const uint32_t segment_bytes = row_segment_bytes(block);

        // Every live row of tile-row i, from its stage.
        auto write_rows = [&](uint32_t i) __attribute__((always_inline)) {
            const uint32_t g = block.tile_row + i;
            const uint32_t live_rows = live_rows_of(g);
            const uint32_t first_page = first_row_page_of(g);
            uint32_t src = get_read_ptr(cb_row_block) + i * Geometry::stage_tile_row_bytes;
            for (uint32_t r = 0; r < live_rows; ++r) {
                noc_async_write(src, output.get_noc_addr(first_page + r, segment_offset), segment_bytes);
                src += Geometry::stage_row_bytes;
            }
        };
        // The live rows of tile-row i in `row_mask` (bit r: row r). A NaN in a padding row (poisoned
        // tile padding) is patched in the stage but never written: its page belongs to another row.
        auto rewrite_rows = [&](uint32_t i, uint32_t row_mask) __attribute__((always_inline)) {
            const uint32_t g = block.tile_row + i;
            const uint32_t live_rows = live_rows_of(g);
            if (live_rows < TILE_HW) {
                row_mask &= (1u << live_rows) - 1;
            }
            const uint32_t first_page = first_row_page_of(g);
            const uint32_t stage = get_read_ptr(cb_row_block) + i * Geometry::stage_tile_row_bytes;
            while (row_mask != 0) {
                const uint32_t r = __builtin_ctz(row_mask);
                row_mask &= row_mask - 1;
                noc_async_write(
                    stage + r * Geometry::stage_row_bytes,
                    output.get_noc_addr(first_page + r, segment_offset),
                    segment_bytes);
            }
        };
        // Tile-row i's rows are already on the wire, unpatched.
        auto check_marks = [&](uint32_t i) __attribute__((always_inline)) {
            cb_wait_front(cb_nan_marks, 1);
            // The marks were stored by the reader's RISC; this one may hold stale lines of the page.
            invalidate_l1_cache();
            const uint32_t page = get_read_ptr(cb_nan_marks);
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page) != 0) {
                DeviceZoneScopedN("repatch_nan_rows");
                noc_async_write_barrier();
                const uint32_t rows = patch_tile_row_nans_rows(
                    page, block.cols, get_read_ptr(cb_row_block) + i * Geometry::stage_tile_row_bytes);
                // Drain the patch stores before the NoC reads the stage.
                invalidate_l1_cache();
                rewrite_rows(i, rows);
            }
            cb_pop_front(cb_nan_marks, 1);
        };

        {
            DeviceZoneScopedN("wait_row_block");
            if constexpr (tiled_stage) {
                cb_wait_front(cb_row_block, Geometry::block_cols_tiles);
            } else {
                cb_wait_front(cb_row_block, 1);
            }
        }
        {
            DeviceZoneScopedN("store_block");
            write_rows(0);
            for (uint32_t i = 1; i < block.rows; ++i) {
                if constexpr (tiled_stage) {
                    cb_wait_front(cb_row_block, (i + 1) * Geometry::block_cols_tiles);
                }
                write_rows(i);
                if constexpr (patch_bfp8_nans) {
                    if (i == 1) {
                        DeviceZoneScopedN("patch_block_nans");
                        check_marks(0);
                    } else {
                        check_marks(i - 1);
                    }
                }
            }
            if constexpr (patch_bfp8_nans) {
                if (block.rows == 1) {
                    DeviceZoneScopedN("patch_block_nans");
                    check_marks(0);
                } else {
                    check_marks(block.rows - 1);
                }
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
