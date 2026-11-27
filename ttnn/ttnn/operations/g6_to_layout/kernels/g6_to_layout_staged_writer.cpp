// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// R1 staged_reorder writer (BRISC / NoC1): store_block.
//
// Per block: one cb_wait_front, one write per live row carrying the block's live bytes of that
// row, one noc_async_writes_flushed, one cb_pop_front. The segment ends at the row's last live
// byte, so stale stage bytes (padding columns, unread faces) never leave L1.
// Rows are addressed through get_noc_addr(page, offset): the RM page stride is the aligned
// page size, not 2 * W.
// Zones (profiler builds only): per block, the wait and the block's work each have a zone; the
// flush inside the work is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr auto output_args = TensorAccessorArgs<Geometry::next_ct_arg + 1>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr);

    DeviceZoneScopedN("g6_staged_writer");
    BlockWalk walk{tile_begin, tile_begin + num_tiles};
    Block block;
    while (walk.next(block)) {
        const uint32_t segment_offset = block.col * TILE_ROW_BYTES;
        const uint32_t segment_end = min_u32(Geometry::width, (block.col + block.cols) * TILE_HW) * DATUM_BYTES;
        const uint32_t segment_bytes = segment_end - segment_offset;

        {
            DeviceZoneScopedN("wait_row_block");
            cb_wait_front(cb_row_block, 1);
        }
        {
            DeviceZoneScopedN("store_block");
            uint32_t tile_row_src = get_read_ptr(cb_row_block);
            for (uint32_t i = 0; i < block.rows; ++i) {
                const uint32_t g = block.tile_row + i;
                const uint32_t live_rows = live_rows_of(g);
                const uint32_t first_page = first_out_page_of(g);
                uint32_t src = tile_row_src;
                for (uint32_t r = 0; r < live_rows; ++r) {
                    noc_async_write(src, output.get_noc_addr(first_page + r, segment_offset), segment_bytes);
                    src += Geometry::stage_row_bytes;
                }
                tile_row_src += Geometry::stage_tile_row_bytes;
            }
            {
                DeviceZoneScopedSumN1("row_block_writes_flushed");
                noc_async_writes_flushed();
            }
        }
        cb_pop_front(cb_row_block, 1);
    }
    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
