// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block writer (BRISC / NoC1): store_block of R1 staged_reorder, packer_untilize and
// sharded_in_place.
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
#include "row_issue.hpp"
namespace row_issue {
constexpr uint32_t G6_ROW_ISSUE = row_issue::ISSUE_CURSOR;
}
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

        {
            DeviceZoneScopedN("wait_row_block");
            cb_wait_front(cb_row_block, block_pages);
        }
        if constexpr (patch_bfp8_nans) {
            DeviceZoneScopedN("patch_block_nans");
            cb_wait_front(cb_nan_marks, 1);
            // The marks were stored by the reader's RISC; this one may hold stale lines of the page.
            invalidate_l1_cache();
            patch_block_nans(get_read_ptr(cb_nan_marks), block, get_read_ptr(cb_row_block));
            cb_pop_front(cb_nan_marks, 1);
            // Drain the patch stores before the NoC reads the stage.
            invalidate_l1_cache();
        }
        {
            DeviceZoneScopedN("store_block");
            uint32_t tile_row_src = get_read_ptr(cb_row_block);
            for (uint32_t i = 0; i < block.rows; ++i) {
                const uint32_t g = block.tile_row + i;
                row_issue::write_rows<row_issue::G6_ROW_ISSUE>(
                    output,
                    first_row_page_of(g),
                    live_rows_of(g),
                    segment_offset,
                    segment_bytes,
                    tile_row_src,
                    Geometry::stage_row_bytes);
                tile_row_src += Geometry::stage_tile_row_bytes;
            }
            {
                DeviceZoneScopedSumN1("row_block_writes_flushed");
                noc_async_writes_flushed();
            }
        }
        cb_pop_front(cb_row_block, block_pages);
    }
    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
