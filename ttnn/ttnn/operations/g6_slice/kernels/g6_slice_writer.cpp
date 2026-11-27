// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice writer (BRISC, NoC1): store_block.
//
// Per block of block_pages staging pages: one wait, every write of the block issued back to back,
// one flush (the reader may overwrite the slot once the writes have left L1), one pop. Output pages
// are this core's contiguous range [core_start_page, core_start_page + core_pages). In row_runs /
// col_realign only the logical rows of a page are written, taken from staging rows
// (i + row_offset_in_tile) mod 32 as whole face-row runs (DRAM writes are alignment-checked mod 16,
// face rows are 32 B). Output padding rows are never written.
//
// Raw NoC instead of a kernel_lib helper: no dataflow helper scatters per-run byte ranges of a CB
// block into DRAM pages through a TensorAccessor.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_slice_common.hpp"

namespace {

using namespace g6_slice;

constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t block_pages = get_compile_time_arg_val(2);
constexpr bool row_runs = get_compile_time_arg_val(3) == 1;
constexpr uint32_t tensor_out_ht_tiles = get_compile_time_arg_val(4);
constexpr uint32_t tensor_out_wt_tiles = get_compile_time_arg_val(5);
constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(6);
constexpr auto out_args = TensorAccessorArgs<7>();

}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t row_offset_in_tile = get_common_arg_val<uint32_t>(1);

    uint32_t out_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    uint32_t th = get_arg_val<uint32_t>(2);
    uint32_t tw = get_arg_val<uint32_t>(3);

    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    // [0]: full tile rows, [1]: the last tile row of an image. Built once, never per page.
    StoreRuns row_class_runs[2];
    if constexpr (row_runs) {
        row_class_runs[0] = make_store_runs<page_bytes>(row_offset_in_tile, TILE_ROWS);
        row_class_runs[1] = make_store_runs<page_bytes>(row_offset_in_tile, rows_last_tile_row);
    }

    const uint32_t num_blocks_this_core = (core_pages + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t remaining = core_pages - block_idx * block_pages;
        const uint32_t block_extent = remaining < block_pages ? remaining : block_pages;

        {
            DeviceZoneScopedN("G6_CB_WAIT");
            cb_wait_front(cb_slice_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6_WRITE_ISSUE");
            uint32_t staging_page = get_read_ptr(cb_slice_pages);
            for (uint32_t k = 0; k < block_extent; ++k) {
                if constexpr (!row_runs) {
                    noc_async_write(staging_page, out.get_noc_addr(out_page), page_bytes);
                } else {
                    const StoreRuns& runs = row_class_runs[th == tensor_out_ht_tiles - 1 ? 1 : 0];
                    const uint64_t page_noc_addr = out.get_noc_addr(out_page);
                    for (uint32_t r = 0; r < runs.count; ++r) {
                        noc_async_write(
                            staging_page + runs.src_offset[r], page_noc_addr + runs.dst_offset[r], runs.bytes[r]);
                    }
                    if (++tw == tensor_out_wt_tiles) {
                        tw = 0;
                        if (++th == tensor_out_ht_tiles) {
                            th = 0;
                        }
                    }
                }
                staging_page += page_bytes;
                ++out_page;
            }
        }
        {
            DeviceZoneScopedN("G6_WRITE_FLUSH");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_slice_pages, block_pages);
    }

    DeviceZoneScopedN("G6_WRITE_BARRIER");
    noc_async_write_barrier();
}
