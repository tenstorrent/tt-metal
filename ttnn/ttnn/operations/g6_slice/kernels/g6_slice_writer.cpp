// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_slice writer (BRISC / NoC1): store_block.
//
// For each block: wait for one block of cb_slice_pages, write every page of the block (whole
// pages, or the output's logical rows as rotated face-row runs), flush, pop. One write barrier at
// the end so no write is in flight when the next program starts.
//
// store_block is raw NoC for the same reason as fetch_block (see the reader).

#include "api/dataflow/dataflow_api.h"
#include "g6_slice_runs.hpp"

void kernel_main() {
    constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t block_pages = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_out_ht_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_out_wt_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(5);
    constexpr bool ROW_RUNS = get_compile_time_arg_val(6) != 0;
    constexpr auto out_args = TensorAccessorArgs<7>();

    using Geometry = g6_slice::TileGeometry<page_bytes>;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t row_offset_in_tile = get_common_arg_val<uint32_t>(1);

    const uint32_t core_start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    uint32_t th = get_arg_val<uint32_t>(2);
    uint32_t tw = get_arg_val<uint32_t>(3);

    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    [[maybe_unused]] g6_slice::RunList store_full, store_last;
    if constexpr (ROW_RUNS) {
        store_full = Geometry::store_runs(row_offset_in_tile, g6_slice::TILE_DIM);
        store_last = Geometry::store_runs(row_offset_in_tile, rows_last_tile_row);
    }

    uint32_t out_page = core_start_page;
    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    uint32_t pages_left = core_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t pages_this_block = pages_left < block_pages ? pages_left : block_pages;
        pages_left -= pages_this_block;

        {
            DeviceZoneScopedN("G6_CB_WAIT");
            cb_wait_front(cb_slice_pages, block_pages);
        }
        // store_block
        {
            DeviceZoneScopedN("G6_WRITE_ISSUE");
            uint32_t page_l1 = get_read_ptr(cb_slice_pages);
            for (uint32_t p = 0; p < pages_this_block; ++p) {
                if constexpr (ROW_RUNS) {
                    const g6_slice::RunList& runs = th == tensor_out_ht_tiles - 1 ? store_last : store_full;
                    for (uint32_t r = 0; r < runs.count; ++r) {
                        const g6_slice::ByteRun& run = runs.runs[r];
                        noc_async_write<page_bytes>(
                            page_l1 + run.src_offset, out.get_noc_addr(out_page, run.dst_offset), run.bytes);
                    }
                    if (++tw == tensor_out_wt_tiles) {
                        tw = 0;
                        if (++th == tensor_out_ht_tiles) {
                            th = 0;
                        }
                    }
                } else {
                    noc_async_write<page_bytes>(page_l1, out.get_noc_addr(out_page), page_bytes);
                }
                page_l1 += page_bytes;
                ++out_page;
            }
        }
        {
            // The reader refills this slot after the pop: the writes must have left L1 first.
            DeviceZoneScopedN("G6_WRITE_FLUSH");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_slice_pages, block_pages);
    }

    DeviceZoneScopedN("G6_WRITE_BARRIER");
    noc_async_write_barrier();
}
