// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice reader (NCRISC, NoC0): fetch_block (+ realign_columns_block in col_realign).
//
// fetch_block is raw NoC: no kernel_lib helper gathers per-page byte runs from DRAM pages into a CB
// block behind one barrier (op_design.md, "Helpers considered and rejected"). realign_columns_block
// uses dataflow_kernel_lib::copy_tile_column.
//
// Each core owns the contiguous output pages [core_start_page, core_start_page + core_pages) and walks
// their (o0, o1, th, tw) coordinates from the host-decomposed start. Every output page reads only the
// input bytes that land in its logical rows:
//   page_copy   one whole input page;
//   row_runs    rows [off, off + nu) of the upper input tile and rows [0, nl) of the tile below it, at
//               their natural in-tile offsets (staging row s holds output row (s - off) mod 32; the
//               writer undoes the rotation);
//   col_realign as row_runs on the face halves holding input columns [c, 32) of the left tile, plus the
//               halves holding [0, c) of the right tile into cb_right_col_tile; the RISC-V then shifts
//               the columns into place before the block is pushed.

#include <stdint.h>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp"
#include "g6_slice_runs.hpp"

namespace {

constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
constexpr uint32_t cb_right_col_tile = get_compile_time_arg_val(1);
constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(3);
constexpr uint32_t block_pages = get_compile_time_arg_val(4);
constexpr bool row_runs = get_compile_time_arg_val(5) != 0;
constexpr bool w_shift = get_compile_time_arg_val(6) != 0;
constexpr uint32_t in_lead1 = get_compile_time_arg_val(7);
constexpr uint32_t out_lead1 = get_compile_time_arg_val(8);
constexpr uint32_t in_ht = get_compile_time_arg_val(9);
constexpr uint32_t in_wt = get_compile_time_arg_val(10);
constexpr uint32_t out_ht = get_compile_time_arg_val(11);
constexpr uint32_t out_wt = get_compile_time_arg_val(12);
constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(13);
constexpr uint32_t cols_last_tile_col = get_compile_time_arg_val(14);
constexpr auto x_args = TensorAccessorArgs<15>();

constexpr uint32_t TILE_COLS = 32;
constexpr uint32_t HALF_COLS = 16;

using elem_t = std::conditional_t<face_row_bytes == 64, uint32_t, uint16_t>;

// Output-page coordinates and the input page of the current output page's upper tile.
struct InputWalk {
    uint32_t o0, o1, th, tw;
    uint32_t lead_begin0, lead_begin1, first_row, first_col;
    uint32_t row_page;  // input page of (o0, o1, th, tw = 0)

    void seek_row() {
        const uint32_t image = (lead_begin0 + o0) * in_lead1 + lead_begin1 + o1;
        row_page = image * (in_ht * in_wt) + (first_row + th) * in_wt + first_col;
    }
    uint32_t upper_page() const { return row_page + tw; }
    bool last_tile_row() const { return th == out_ht - 1; }
    void advance() {
        if (++tw < out_wt) {
            return;
        }
        tw = 0;
        if (++th == out_ht) {
            th = 0;
            if (++o1 == out_lead1) {
                o1 = 0;
                ++o0;
            }
        }
        seek_row();
    }
};

// Run lists of one row class (tile rows holding 32 logical rows, or the last tile row of an image).
struct FetchRuns {
    g6_slice::RunList upper, lower;              // left / only input tile column
    g6_slice::RunList upper_right, lower_right;  // col_realign: the input tile column to the right
};

FetchRuns full_rows;
FetchRuns last_rows;

FetchRuns fetch_runs(uint32_t off, uint32_t rows, uint32_t col_off) {
    const uint32_t nu = rows < g6_slice::TILE_ROWS - off ? rows : g6_slice::TILE_ROWS - off;
    const uint32_t nl = rows - nu;
    const uint32_t left_halves = (w_shift && col_off >= HALF_COLS) ? 0b10 : g6_slice::BOTH_HALVES;
    const uint32_t right_halves = col_off > HALF_COLS ? g6_slice::BOTH_HALVES : 0b01;
    FetchRuns runs{};
    runs.upper = g6_slice::band_runs<face_row_bytes>(off, off + nu, left_halves);
    runs.lower = g6_slice::band_runs<face_row_bytes>(0, nl, left_halves);
    if constexpr (w_shift) {
        runs.upper_right = g6_slice::band_runs<face_row_bytes>(off, off + nu, right_halves);
        runs.lower_right = g6_slice::band_runs<face_row_bytes>(0, nl, right_halves);
    }
    return runs;
}

FORCE_INLINE uint32_t logical_cols(uint32_t tw) { return tw == out_wt - 1 ? cols_last_tile_col : TILE_COLS; }

FORCE_INLINE void issue_reads(const g6_slice::RunList& runs, uint64_t src_page, uint32_t dst_page) {
    for (uint32_t k = 0; k < runs.count; ++k) {
        const g6_slice::PageRun& run = runs.runs[k];
        noc_async_read<page_bytes>(src_page + run.src, dst_page + run.dst, run.len);
    }
}

template <typename Accessor>
void fetch_block(
    const Accessor& x, InputWalk& walk, uint32_t num_pages, uint32_t slot_l1, uint32_t right_l1, uint32_t col_off) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        const uint32_t dst = slot_l1 + i * page_bytes;
        const uint32_t upper = walk.upper_page();
        if constexpr (!row_runs) {
            noc_async_read<page_bytes>(x.get_noc_addr(upper), dst, page_bytes);
        } else {
            const FetchRuns& runs = walk.last_tile_row() ? last_rows : full_rows;
            issue_reads(runs.upper, x.get_noc_addr(upper), dst);
            if (runs.lower.count != 0) {
                issue_reads(runs.lower, x.get_noc_addr(upper + in_wt), dst);
            }
            if constexpr (w_shift) {
                if (logical_cols(walk.tw) > TILE_COLS - col_off) {
                    const uint32_t right_dst = right_l1 + i * page_bytes;
                    issue_reads(runs.upper_right, x.get_noc_addr(upper + 1), right_dst);
                    if (runs.lower_right.count != 0) {
                        issue_reads(runs.lower_right, x.get_noc_addr(upper + 1 + in_wt), right_dst);
                    }
                }
            }
        }
        walk.advance();
    }
}

// Column j <- staging column j + c, or right-tile column j + c - 32. Ascending j is in place-safe:
// column j + c >= j is read before it is overwritten.
void realign_columns_block(uint32_t tw, uint32_t num_pages, uint32_t slot_l1, uint32_t right_l1, uint32_t col_off) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        const uint32_t page = slot_l1 + i * page_bytes;
        const uint32_t right = right_l1 + i * page_bytes;
        const uint32_t cols = logical_cols(tw);
        for (uint32_t j = 0; j < cols; ++j) {
            if (j + col_off < TILE_COLS) {
                dataflow_kernel_lib::copy_tile_column<cb_slice_pages, elem_t>(page, j + col_off, page, j);
            } else {
                dataflow_kernel_lib::copy_tile_column<cb_slice_pages, elem_t>(right, j + col_off - TILE_COLS, page, j);
            }
        }
        if (++tw == out_wt) {
            tw = 0;
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t lead_begin0 = get_common_arg_val<uint32_t>(1);
    const uint32_t lead_begin1 = get_common_arg_val<uint32_t>(2);
    const uint32_t first_row = get_common_arg_val<uint32_t>(3);
    const uint32_t row_off = get_common_arg_val<uint32_t>(4);
    const uint32_t first_col = get_common_arg_val<uint32_t>(5);
    const uint32_t col_off = get_common_arg_val<uint32_t>(6);

    const uint32_t core_pages = get_arg_val<uint32_t>(0);
    InputWalk walk{
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        lead_begin0,
        lead_begin1,
        first_row,
        first_col,
        0};
    walk.seek_row();

    const auto x = TensorAccessor(x_args, x_addr, page_bytes);

    if constexpr (row_runs) {
        full_rows = fetch_runs(row_off, g6_slice::TILE_ROWS, col_off);
        last_rows = fetch_runs(row_off, rows_last_tile_row, col_off);
    }
    uint32_t right_l1 = 0;
    if constexpr (w_shift) {
        right_l1 = get_write_ptr(cb_right_col_tile);
    }

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t done = block_idx * block_pages;
        const uint32_t num_pages = core_pages - done < block_pages ? core_pages - done : block_pages;
        {
            DeviceZoneScopedN("G6_CB_RESERVE");
            cb_reserve_back(cb_slice_pages, block_pages);
        }
        const uint32_t slot_l1 = get_write_ptr(cb_slice_pages);
        const uint32_t block_tw = walk.tw;
        {
            DeviceZoneScopedN("G6_FETCH_ISSUE");
            fetch_block(x, walk, num_pages, slot_l1, right_l1, col_off);
        }
        {
            DeviceZoneScopedN("G6_READ_BARRIER");
            noc_async_read_barrier();
        }
        if constexpr (w_shift) {
            DeviceZoneScopedN("G6_COL_REALIGN");
            realign_columns_block(block_tw, num_pages, slot_l1, right_l1, col_off);
        }
        // Nominal count even on a ragged last block: every push is block_pages, so the ring of
        // cb_depth * block_pages pages always closes.
        cb_push_back(cb_slice_pages, block_pages);
    }
}
