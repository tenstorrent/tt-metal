// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block writer (BRISC): store_block of R1 staged_reorder, packer_untilize and sharded_in_place.
//
// bf8b_tile_row_pipeline (recommended form):
//  - A tiled stage (R3/R4) is consumed TILE-ROW by tile-row: the writer waits tile-row i's untilized
//    pages (cumulative cb_wait_front on the block's pages; the block is still flushed and popped once)
//    and writes its live rows at once. The NaN marks never gate a write: tile-row i's cb_nan_marks page
//    (one per live tile-row, g6_to_layout_bfp8_nan_marks.hpp) is checked right after its writes are
//    issued. Clean tile-row (summary word 0, the common case): one wait, one fence, one load, one pop.
//    A tile-row holding a NaN datum: write barrier (its unpatched rows land first), store 0xFFFF over
//    the marked datums in the stage, fence, re-write only its LIVE rows that hold one (a padding row's
//    page belongs to another row). The stage is popped after the block's writes are flushed, so it is
//    intact for any re-write.
//  - G6_CHECKER_NOC (tiled stage only): cores whose physical NoC0 x + y is odd write on NoC0, the others
//    on NoC1, so the all-to-all row writes of a regime are spread over both NoCs (on c37 the bottom rows'
//    NoC1 writes otherwise run 35-40% slower than the median core). In dedicated NoC mode only the
//    kernel's own NoC counters are initialized, so the writer syncs its NoC0 counters at start. Valid
//    while no other RISC of the core issues NoC0 writes during the kernel (R3/R4 readers only read).
//  - Every lambda is forced inline and no look-ahead block state is kept: an out-of-line lambda or extra
//    live state spills the row-write loop's invariants (6-7 stack reloads per write vs 2 measured, c43
//    15.6 us vs 13.1 us).
// Rows are addressed through get_noc_addr(page, offset, noc): the RM page stride is the aligned page size.
// Zones (profiler builds only): per block wait_row_block (tile-row 0's pages) and store_block (the
// rest); the flush is a sum zone (TT_METAL_PROFILER_SUM).

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"
#include "g6_to_layout_bfp8_nan_marks.hpp"

using namespace g6_to_layout;

#ifndef G6_CHECKER_NOC
#define G6_CHECKER_NOC 0
#endif

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
    // The whole block loop is instantiated once per NoC (compile-time NoC index): a runtime NoC index in
    // the row-write loop costs ~8 cycles per write (c48: 62 vs 54 cycles per 256-B write).
    auto run = [&](auto noc_tag) __attribute__((always_inline)) {
        constexpr uint8_t wnoc = decltype(noc_tag)::value;
        CoreBlocks<sharded> blocks;
        Block block;
        while (blocks.next(block)) {
            const uint32_t block_pages = tiled_stage ? moved_tile_rows<sharded>(block) * Geometry::block_cols_tiles : 1;
            const uint32_t segment_offset = block.col * TILE_ROW_BYTES;
            const uint32_t segment_bytes = row_segment_bytes(block);

            auto stage_of = [&](uint32_t i) __attribute__((always_inline)) {
                return get_read_ptr(cb_row_block) + i * Geometry::stage_tile_row_bytes;
            };
            auto write_rows = [&](uint32_t i) __attribute__((always_inline)) {
                const uint32_t g = block.tile_row + i;
                const uint32_t live_rows = live_rows_of(g);
                const uint32_t first_page = first_row_page_of(g);
                uint32_t src = stage_of(i);
                for (uint32_t r = 0; r < live_rows; ++r) {
                    noc_async_write(
                        src, output.get_noc_addr(first_page + r, segment_offset, wnoc), segment_bytes, wnoc);
                    src += Geometry::stage_row_bytes;
                }
            };
            // Tile-row i's rows are already on the wire, unpatched.
            auto check_marks = [&](uint32_t i) __attribute__((always_inline)) {
                cb_wait_front(cb_nan_marks, 1);
                // The marks were stored by the reader's RISC; this one may hold stale lines of the page.
                invalidate_l1_cache();
                const uint32_t page = get_read_ptr(cb_nan_marks);
                if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page) != 0) {
                    noc_async_write_barrier(wnoc);
                    uint32_t rows = patch_tile_row_nans_rows(page, block.cols, stage_of(i));
                    // Drain the patch stores before the NoC reads the stage.
                    invalidate_l1_cache();
                    const uint32_t g = block.tile_row + i;
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
                            output.get_noc_addr(first_page + r, segment_offset, wnoc),
                            segment_bytes,
                            wnoc);
                    }
                }
                cb_pop_front(cb_nan_marks, 1);
            };

            {
                DeviceZoneScopedN("wait_row_block");
                cb_wait_front(cb_row_block, tiled_stage ? Geometry::block_cols_tiles : 1);
            }
            {
                DeviceZoneScopedN("store_block");
                for (uint32_t i = 0; i < block.rows; ++i) {
                    if constexpr (tiled_stage) {
                        if (i > 0) {
                            cb_wait_front(cb_row_block, (i + 1) * Geometry::block_cols_tiles);
                        }
                    }
                    write_rows(i);
                    if constexpr (patch_bfp8_nans) {
                        check_marks(i);
                    }
                }
                {
                    DeviceZoneScopedSumN1("row_block_writes_flushed");
                    noc_async_writes_flushed(wnoc);
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
            noc_async_write_barrier(wnoc);
        }
    };
    constexpr uint8_t other_noc = 1 - noc_index;
    if constexpr (G6_CHECKER_NOC != 0 && tiled_stage) {
        if ((my_x[0] + my_y[0]) & 1) {
            noc_local_state_init(other_noc);
            run(std::integral_constant<uint8_t, other_noc>{});
            return;
        }
    }
    run(std::integral_constant<uint8_t, noc_index>{});
}
