// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice writer (BRISC, NoC1): store_block.
//
// store_block is raw NoC: no kernel_lib helper scatters per-page byte runs from a CB block to DRAM
// pages (op_design.md, "Helpers considered and rejected").
//
// page_copy writes each staging page whole. row_runs / col_realign write only the logical rows
// [0, rv) of each output page, taking output row i from staging row (i + off) mod 32; every run is a
// whole number of face rows, so source and destination stay congruent mod 16 (DRAM write alignment).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "g6_slice_runs.hpp"

namespace {

constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(2);
constexpr uint32_t block_pages = get_compile_time_arg_val(3);
constexpr bool row_runs = get_compile_time_arg_val(4) != 0;
constexpr uint32_t out_ht = get_compile_time_arg_val(5);
constexpr uint32_t out_wt = get_compile_time_arg_val(6);
constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(7);
constexpr auto out_args = TensorAccessorArgs<8>();

g6_slice::RunList full_rows;
g6_slice::RunList last_rows;

// Tile-row position of the current output page; only the row class (last tile row or not) matters.
struct RowWalk {
    uint32_t th, tw;
    bool last_tile_row() const { return th == out_ht - 1; }
    void advance() {
        if (++tw == out_wt) {
            tw = 0;
            if (++th == out_ht) {
                th = 0;
            }
        }
    }
};

template <typename Accessor>
void store_block(const Accessor& out, RowWalk& walk, uint32_t first_page, uint32_t num_pages, uint32_t slot_l1) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        const uint32_t src = slot_l1 + i * page_bytes;
        const uint64_t dst = out.get_noc_addr(first_page + i);
        if constexpr (!row_runs) {
            noc_async_write<page_bytes>(src, dst, page_bytes);
        } else {
            const g6_slice::RunList& runs = walk.last_tile_row() ? last_rows : full_rows;
            for (uint32_t k = 0; k < runs.count; ++k) {
                const g6_slice::PageRun& run = runs.runs[k];
                noc_async_write<page_bytes>(src + run.src, dst + run.dst, run.len);
            }
            walk.advance();
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t row_off = get_common_arg_val<uint32_t>(1);

    const uint32_t core_start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    RowWalk walk{get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};

    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    if constexpr (row_runs) {
        full_rows = g6_slice::rotate_runs<face_row_bytes>(row_off, g6_slice::TILE_ROWS);
        last_rows = g6_slice::rotate_runs<face_row_bytes>(row_off, rows_last_tile_row);
    }

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t done = block_idx * block_pages;
        const uint32_t num_pages = core_pages - done < block_pages ? core_pages - done : block_pages;
        {
            DeviceZoneScopedN("G6_CB_WAIT");
            cb_wait_front(cb_slice_pages, block_pages);
        }
        {
            DeviceZoneScopedN("G6_WRITE_ISSUE");
            store_block(out, walk, core_start_page + done, num_pages, get_read_ptr(cb_slice_pages));
        }
        {
            // The slot is refilled by the reader as soon as it is popped; its bytes must have left L1.
            DeviceZoneScopedN("G6_WRITE_FLUSH");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_slice_pages, block_pages);
    }
    {
        DeviceZoneScopedN("G6_WRITE_BARRIER");
        noc_async_write_barrier();
    }
}
