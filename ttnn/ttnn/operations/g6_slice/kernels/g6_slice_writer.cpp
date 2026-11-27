// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice writer (BRISC, NoC1): store_block.
//
// Raw NoC instead of a kernel_lib helper: no dataflow helper scatters per-page byte-run lists from a CB
// block into interleaved DRAM pages (op_design.md, API Mapping).
//
// Output pages of this core are contiguous from core_start_page. page_copy writes each staging page
// whole. row_runs / col_realign write only the logical rows [0, rv) of each page, un-rotating the
// staging page (staging row (i + off) mod 32 -> output row i); padding rows of out are never written.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_slice_runs.hpp"

void kernel_main() {
    constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t block_pages = get_compile_time_arg_val(2);
    constexpr bool row_runs = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t tensor_out_ht_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_out_wt_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();

    constexpr uint32_t face_row_bytes = page_bytes / 64;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t row_offset_in_tile = get_common_arg_val<uint32_t>(1);

    uint32_t out_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    uint32_t th = get_arg_val<uint32_t>(2);
    uint32_t tw = get_arg_val<uint32_t>(3);

    const auto out_accessor = TensorAccessor(out_args, out_addr, page_bytes);

    g6_slice::RunList<8> full_rows{}, last_rows{};
    if constexpr (row_runs) {
        full_rows = g6_slice::rotated_runs<face_row_bytes>(row_offset_in_tile, g6_slice::TILE_ROWS);
        last_rows = g6_slice::rotated_runs<face_row_bytes>(row_offset_in_tile, rows_last_tile_row);
    }

    uint32_t remaining = core_pages;
    while (remaining > 0) {
        const uint32_t pages_this_block = remaining < block_pages ? remaining : block_pages;
        {
            DeviceZoneScopedN("G6_CB_WAIT");
            cb_wait_front(cb_slice_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6_WRITE_ISSUE");
            uint32_t l1_page = get_read_ptr(cb_slice_pages);
            for (uint32_t p = 0; p < pages_this_block; ++p) {
                const uint64_t page_noc_addr = out_accessor.get_noc_addr(out_page);
                if constexpr (!row_runs) {
                    noc_async_write(l1_page, page_noc_addr, page_bytes);
                } else {
                    const auto& rows = th == tensor_out_ht_tiles - 1 ? last_rows : full_rows;
                    for (uint32_t i = 0; i < rows.count; ++i) {
                        noc_async_write(
                            l1_page + rows.runs[i].src_offset,
                            page_noc_addr + rows.runs[i].dst_offset,
                            rows.runs[i].size);
                    }
                    if (++tw == tensor_out_wt_tiles) {
                        tw = 0;
                        if (++th == tensor_out_ht_tiles) {
                            th = 0;
                        }
                    }
                }
                ++out_page;
                l1_page += page_bytes;
            }
        }
        {
            // The reader refills this slot right after the pop: the writes must have left L1 first.
            DeviceZoneScopedN("G6_WRITE_FLUSH");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_slice_pages, block_pages);
        remaining -= pages_this_block;
    }
    {
        DeviceZoneScopedN("G6_WRITE_BARRIER");
        noc_async_write_barrier();
    }
}
