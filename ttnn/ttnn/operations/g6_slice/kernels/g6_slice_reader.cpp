// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice reader (NCRISC, NoC0): fetch_block (+ realign_columns_block in col_realign).
//
// Walks this core's contiguous output-page range in (lead0, lead1, tile row, tile col) order and,
// per block of block_pages output pages, issues every read of the block back to back, waits on one
// barrier and pushes the block. Each read lands at its natural in-tile byte offset (DRAM reads are
// alignment-checked mod 64 against the L1 destination), so a staging page holds output row i at
// staging row (i + row_offset_in_tile) mod 32; the writer undoes that rotation.
//
// Raw NoC instead of a kernel_lib helper: no dataflow helper gathers per-run byte ranges of DRAM
// pages from a TensorAccessor into a CB block (the closest, read_with_state, is self-aimed L1 with
// one fixed size). A "page-run gather" helper taking an accessor, a page sequence and a run list
// would own fetch_block.

#include <stdint.h>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp"
#include "g6_slice_common.hpp"

namespace {

using namespace g6_slice;

constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
constexpr uint32_t cb_right_col_tile = get_compile_time_arg_val(1);
constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
constexpr uint32_t block_pages = get_compile_time_arg_val(3);
constexpr bool row_runs = get_compile_time_arg_val(4) == 1;
constexpr bool w_shift = get_compile_time_arg_val(5) == 1;
constexpr uint32_t tensor_in_ht_tiles = get_compile_time_arg_val(6);
constexpr uint32_t tensor_in_wt_tiles = get_compile_time_arg_val(7);
constexpr uint32_t tensor_out_ht_tiles = get_compile_time_arg_val(8);
constexpr uint32_t tensor_out_wt_tiles = get_compile_time_arg_val(9);
constexpr uint32_t tensor_in_lead1 = get_compile_time_arg_val(10);
constexpr uint32_t tensor_out_lead1 = get_compile_time_arg_val(11);
constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(12);
constexpr uint32_t cols_last_tile_col = get_compile_time_arg_val(13);
constexpr auto x_args = TensorAccessorArgs<14>();

constexpr uint32_t tensor_in_image_pages = tensor_in_ht_tiles * tensor_in_wt_tiles;

// bfloat16 tiles move 16-bit columns; a 32-bit T would swap column pairs.
static_assert(!w_shift || page_bytes == 2 * TILE_ROWS * TILE_COLS || page_bytes == 4 * TILE_ROWS * TILE_COLS);
using Element = std::conditional_t<page_bytes == 2 * TILE_ROWS * TILE_COLS, uint16_t, uint32_t>;

// Row bands of one row class (all rows, or the rows of the last tile row of an image). The left
// band runs fetch the input tile column the output tile starts in; the right ones (col_realign
// only) the next input tile column, which supplies output columns [32 - c, 32).
struct RowClassRuns {
    BandRuns upper;
    BandRuns lower;
    BandRuns right_upper;
    BandRuns right_lower;
};

RowClassRuns make_row_class_runs(
    uint32_t row_offset_in_tile, uint32_t rows_in_tile, uint32_t left_mask, uint32_t right_mask) {
    const uint32_t rows_upper =
        rows_in_tile < TILE_ROWS - row_offset_in_tile ? rows_in_tile : TILE_ROWS - row_offset_in_tile;
    const uint32_t rows_lower = rows_in_tile - rows_upper;
    RowClassRuns runs;
    runs.upper = make_band_runs<page_bytes>(row_offset_in_tile, row_offset_in_tile + rows_upper, left_mask);
    runs.lower = make_band_runs<page_bytes>(0, rows_lower, left_mask);
    if constexpr (w_shift) {
        runs.right_upper = make_band_runs<page_bytes>(row_offset_in_tile, row_offset_in_tile + rows_upper, right_mask);
        runs.right_lower = make_band_runs<page_bytes>(0, rows_lower, right_mask);
    }
    return runs;
}

// Output page (o0, o1, th, tw) <- input pages from tile row (first_input_tile_row + th) and, for a
// shifted row band, the tile row below it.
struct InputWalk {
    uint32_t lead_begin0;
    uint32_t lead_begin1;
    uint32_t first_input_tile_row;
    uint32_t first_input_tile_col;
    uint32_t o0;
    uint32_t o1;
    uint32_t th;
    uint32_t tw;
    uint32_t tile_row_page;  // input page of (image, first_input_tile_row + th, first_input_tile_col)

    void seek_tile_row() {
        const uint32_t image = (lead_begin0 + o0) * tensor_in_lead1 + lead_begin1 + o1;
        tile_row_page =
            image * tensor_in_image_pages + (first_input_tile_row + th) * tensor_in_wt_tiles + first_input_tile_col;
    }

    uint32_t upper_page() const { return tile_row_page + tw; }
    bool last_tile_row() const { return th == tensor_out_ht_tiles - 1; }
    bool last_tile_col() const { return tw == tensor_out_wt_tiles - 1; }

    void advance() {
        if (++tw < tensor_out_wt_tiles) {
            return;
        }
        tw = 0;
        if (++th < tensor_out_ht_tiles) {
            tile_row_page += tensor_in_wt_tiles;
            return;
        }
        th = 0;
        if (++o1 == tensor_out_lead1) {
            o1 = 0;
            ++o0;
        }
        seek_tile_row();
    }
};

template <typename Accessor>
FORCE_INLINE void read_band(const Accessor& x, uint32_t page, const BandRuns& runs, uint32_t staging_page) {
    if (runs.count == 0) {
        return;
    }
    const uint64_t page_noc_addr = x.get_noc_addr(page);
    for (uint32_t r = 0; r < runs.count; ++r) {
        noc_async_read(page_noc_addr + runs.offset[r], staging_page + runs.offset[r], runs.bytes[r]);
    }
}

// In place, ascending: column j <- staging column j + c, or right-tile column j + c - 32. Column
// j + c >= j is read before it is overwritten. Must run after the read barrier, which invalidates
// the L1 cache, so the RISC-V loads see the NoC-written bytes.
void realign_columns_block(uint32_t staging_block, uint32_t right_block, uint32_t pages, uint32_t col_offset_in_tile) {
    for (uint32_t k = 0; k < pages; ++k) {
        const uint32_t staging_page = staging_block + k * page_bytes;
        const uint32_t right_page = right_block + k * page_bytes;
        for (uint32_t col = 0; col < TILE_COLS; ++col) {
            const uint32_t src_col = col + col_offset_in_tile;
            if (src_col < TILE_COLS) {
                dataflow_kernel_lib::copy_tile_column<cb_slice_pages, Element>(
                    staging_page, src_col, staging_page, col);
            } else {
                dataflow_kernel_lib::copy_tile_column<cb_slice_pages, Element>(
                    right_page, src_col - TILE_COLS, staging_page, col);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    InputWalk walk;
    walk.lead_begin0 = get_common_arg_val<uint32_t>(1);
    walk.lead_begin1 = get_common_arg_val<uint32_t>(2);
    walk.first_input_tile_row = get_common_arg_val<uint32_t>(3);
    const uint32_t row_offset_in_tile = get_common_arg_val<uint32_t>(4);
    walk.first_input_tile_col = get_common_arg_val<uint32_t>(5);
    const uint32_t col_offset_in_tile = get_common_arg_val<uint32_t>(6);

    const uint32_t core_pages = get_arg_val<uint32_t>(0);
    walk.o0 = get_arg_val<uint32_t>(1);
    walk.o1 = get_arg_val<uint32_t>(2);
    walk.th = get_arg_val<uint32_t>(3);
    walk.tw = get_arg_val<uint32_t>(4);
    walk.seek_tile_row();

    const auto x = TensorAccessor(x_args, x_addr, page_bytes);

    // Left input tile: columns [c, 32); right input tile: columns [0, c).
    const uint32_t left_mask = col_offset_in_tile < FACE_ROWS ? BOTH_HALVES : RIGHT_HALF;
    const uint32_t right_mask = col_offset_in_tile > FACE_ROWS ? BOTH_HALVES : LEFT_HALF;
    // [0]: full tile rows, [1]: the last tile row of an image. Built once, never per page.
    RowClassRuns row_class_runs[2];
    if constexpr (row_runs) {
        row_class_runs[0] = make_row_class_runs(row_offset_in_tile, TILE_ROWS, left_mask, right_mask);
        row_class_runs[1] = make_row_class_runs(row_offset_in_tile, rows_last_tile_row, left_mask, right_mask);
    }
    // A full output tile column always reaches into the next input tile (c > 0); the last one only
    // when its logical columns reach past 32 - c.
    const bool right_needed_last_col = cols_last_tile_col > TILE_COLS - col_offset_in_tile;

    const uint32_t num_blocks_this_core = (core_pages + block_pages - 1) / block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t remaining = core_pages - block_idx * block_pages;
        const uint32_t block_extent = remaining < block_pages ? remaining : block_pages;

        // fetch_block: push/pop stay at the nominal block_pages so the ring closes at
        // cb_depth * block_pages; a ragged last block narrows only the NoC work.
        {
            DeviceZoneScopedN("G6_CB_RESERVE");
            cb_reserve_back(cb_slice_pages, block_pages);
        }
        const uint32_t staging_block = get_write_ptr(cb_slice_pages);
        const uint32_t right_block = w_shift ? get_write_ptr(cb_right_col_tile) : 0;
        {
            DeviceZoneScopedN("G6_FETCH_ISSUE");
            uint32_t staging_page = staging_block;
            uint32_t right_page = right_block;
            for (uint32_t k = 0; k < block_extent; ++k) {
                const uint32_t upper_page = walk.upper_page();
                if constexpr (!row_runs) {
                    noc_async_read(x.get_noc_addr(upper_page), staging_page, page_bytes);
                } else {
                    const RowClassRuns& runs = row_class_runs[walk.last_tile_row() ? 1 : 0];
                    read_band(x, upper_page, runs.upper, staging_page);
                    read_band(x, upper_page + tensor_in_wt_tiles, runs.lower, staging_page);
                    if constexpr (w_shift) {
                        if (!walk.last_tile_col() || right_needed_last_col) {
                            read_band(x, upper_page + 1, runs.right_upper, right_page);
                            read_band(x, upper_page + 1 + tensor_in_wt_tiles, runs.right_lower, right_page);
                        }
                    }
                }
                staging_page += page_bytes;
                right_page += page_bytes;
                walk.advance();
            }
        }
        {
            DeviceZoneScopedN("G6_READ_BARRIER");
            noc_async_read_barrier();
        }
        if constexpr (w_shift) {
            DeviceZoneScopedN("G6_COL_REALIGN");
            realign_columns_block(staging_block, right_block, block_extent, col_offset_in_tile);
        }
        cb_push_back(cb_slice_pages, block_pages);
    }
}
