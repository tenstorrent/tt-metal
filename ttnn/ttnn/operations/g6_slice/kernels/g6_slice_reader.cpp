// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_slice reader (NCRISC, NoC0): fetch_block + realign_columns_block.
//
// Raw NoC instead of a kernel_lib helper for fetch_block: no dataflow helper gathers per-page byte-run
// lists from interleaved DRAM pages into a CB block under one barrier (op_design.md, API Mapping).
// realign_columns_block uses dataflow_kernel_lib::copy_tile_column.
//
// Each output page (o0, o1, th, tw) of this core's contiguous range is fetched into a staging page of
// cb_slice_pages. page_copy: one whole input page. row_runs: the upper band (rows [off, off + nu) of
// input tile row q + th) and the lower band (rows [0, nl) of the tile row below) land at their natural
// in-tile positions, so staging row s holds output row (s - off) mod 32 and the writer un-rotates.
// col_realign additionally fetches the right input tile column into cb_right_col_tile and shifts
// columns left by c in place after the block's read barrier.

#include <stdint.h>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp"
#include "g6_slice_runs.hpp"

namespace {

using g6_slice::band_runs;
using g6_slice::RunList;
using g6_slice::TILE_COLS;
using g6_slice::TILE_ROWS;

struct ReadRows {
    RunList<4> upper;
    RunList<4> lower;
};

template <uint32_t face_row_bytes>
ReadRows read_rows(uint32_t row_offset, uint32_t rows, uint32_t half_mask) {
    const uint32_t upper_rows = (TILE_ROWS - row_offset) < rows ? (TILE_ROWS - row_offset) : rows;
    return ReadRows{
        band_runs<face_row_bytes>(row_offset, row_offset + upper_rows, half_mask),
        band_runs<face_row_bytes>(0, rows - upper_rows, half_mask)};
}

template <typename Accessor, uint32_t capacity>
FORCE_INLINE void issue_runs(
    const Accessor& accessor, uint32_t page_id, const RunList<capacity>& list, uint32_t l1_page) {
    const uint64_t page_noc_addr = accessor.get_noc_addr(page_id);
    for (uint32_t i = 0; i < list.count; ++i) {
        noc_async_read(page_noc_addr + list.runs[i].src_offset, l1_page + list.runs[i].dst_offset, list.runs[i].size);
    }
}

template <uint32_t in_wt, typename Accessor>
FORCE_INLINE void issue_rows(const Accessor& accessor, uint32_t upper_page, const ReadRows& rows, uint32_t l1_page) {
    issue_runs(accessor, upper_page, rows.upper, l1_page);
    if (rows.lower.count > 0) {
        issue_runs(accessor, upper_page + in_wt, rows.lower, l1_page);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_right_col_tile = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_pages = get_compile_time_arg_val(4);
    constexpr bool row_runs = get_compile_time_arg_val(5) != 0;
    constexpr bool w_shift = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t tensor_in_ht_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_in_wt_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tensor_out_ht_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t tensor_out_wt_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t in_lead1 = get_compile_time_arg_val(11);
    constexpr uint32_t out_lead1 = get_compile_time_arg_val(12);
    constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(13);
    constexpr uint32_t cols_last_tile_col = get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();

    constexpr uint32_t face_row_bytes = page_bytes / 64;
    constexpr uint32_t in_pages_per_image = tensor_in_ht_tiles * tensor_in_wt_tiles;
    // Input-page jumps when the output walk carries out of a tile row, an image, a lead1 run.
    constexpr uint32_t tile_row_skip = tensor_in_wt_tiles - tensor_out_wt_tiles;
    constexpr uint32_t image_skip = (tensor_in_ht_tiles - tensor_out_ht_tiles) * tensor_in_wt_tiles;
    constexpr uint32_t lead1_skip = (in_lead1 - out_lead1) * in_pages_per_image;

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t lead_begin0 = get_common_arg_val<uint32_t>(1);
    const uint32_t lead_begin1 = get_common_arg_val<uint32_t>(2);
    const uint32_t first_input_tile_row = get_common_arg_val<uint32_t>(3);
    const uint32_t row_offset_in_tile = get_common_arg_val<uint32_t>(4);
    const uint32_t first_input_tile_col = get_common_arg_val<uint32_t>(5);
    const uint32_t col_offset_in_tile = get_common_arg_val<uint32_t>(6);

    const uint32_t core_pages = get_arg_val<uint32_t>(0);
    const uint32_t o0 = get_arg_val<uint32_t>(1);
    uint32_t o1 = get_arg_val<uint32_t>(2);
    uint32_t th = get_arg_val<uint32_t>(3);
    uint32_t tw = get_arg_val<uint32_t>(4);

    const auto x_accessor = TensorAccessor(x_args, x_addr, page_bytes);

    uint32_t in_page = ((lead_begin0 + o0) * in_lead1 + lead_begin1 + o1) * in_pages_per_image +
                       (first_input_tile_row + th) * tensor_in_wt_tiles + first_input_tile_col + tw;

    // Row classes: rows = 32, and rows_last_tile_row on the last tile row of every image.
    ReadRows full_left{}, last_left{}, full_right{}, last_right{};
    if constexpr (row_runs) {
        // The left input column supplies columns [c, 32), the right one [0, c): fetch only the face
        // halves holding them.
        const uint32_t left_mask = (w_shift && col_offset_in_tile >= 16) ? 0b10 : g6_slice::BOTH_HALVES;
        full_left = read_rows<face_row_bytes>(row_offset_in_tile, TILE_ROWS, left_mask);
        last_left = read_rows<face_row_bytes>(row_offset_in_tile, rows_last_tile_row, left_mask);
        if constexpr (w_shift) {
            const uint32_t right_mask = col_offset_in_tile > 16 ? g6_slice::BOTH_HALVES : 0b01;
            full_right = read_rows<face_row_bytes>(row_offset_in_tile, TILE_ROWS, right_mask);
            last_right = read_rows<face_row_bytes>(row_offset_in_tile, rows_last_tile_row, right_mask);
        }
    }

    [[maybe_unused]] const uint32_t right_base = w_shift ? get_write_ptr(cb_right_col_tile) : 0;
    auto logical_cols = [&](uint32_t col_tile) {
        return col_tile == tensor_out_wt_tiles - 1 ? cols_last_tile_col : TILE_COLS;
    };

    uint32_t remaining = core_pages;
    while (remaining > 0) {
        // A ragged last block narrows only the NoC work; the handshake stays block_pages so the CB ring
        // (capacity cb_depth x block_pages) never splits a block.
        const uint32_t pages_this_block = remaining < block_pages ? remaining : block_pages;
        {
            DeviceZoneScopedN("G6_CB_RESERVE");
            cb_reserve_back(cb_slice_pages, block_pages);
        }
        const uint32_t block_base = get_write_ptr(cb_slice_pages);
        [[maybe_unused]] const uint32_t block_first_tw = tw;
        {
            DeviceZoneScopedN("G6_FETCH_ISSUE");
            uint32_t l1_page = block_base;
            [[maybe_unused]] uint32_t l1_right = right_base;
            for (uint32_t p = 0; p < pages_this_block; ++p) {
                if constexpr (!row_runs) {
                    noc_async_read(x_accessor.get_noc_addr(in_page), l1_page, page_bytes);
                } else {
                    const bool last_row = th == tensor_out_ht_tiles - 1;
                    issue_rows<tensor_in_wt_tiles>(x_accessor, in_page, last_row ? last_left : full_left, l1_page);
                    if constexpr (w_shift) {
                        if (logical_cols(tw) > TILE_COLS - col_offset_in_tile) {
                            issue_rows<tensor_in_wt_tiles>(
                                x_accessor, in_page + 1, last_row ? last_right : full_right, l1_right);
                        }
                        l1_right += page_bytes;
                    }
                }
                l1_page += page_bytes;

                ++in_page;
                if (++tw == tensor_out_wt_tiles) {
                    tw = 0;
                    in_page += tile_row_skip;
                    if (++th == tensor_out_ht_tiles) {
                        th = 0;
                        in_page += image_skip;
                        if (++o1 == out_lead1) {
                            o1 = 0;
                            in_page += lead1_skip;
                        }
                    }
                }
            }
        }
        {
            DeviceZoneScopedN("G6_READ_BARRIER");
            noc_async_read_barrier();
        }
        if constexpr (w_shift) {
            DeviceZoneScopedN("G6_COL_REALIGN");
            using Elem = std::conditional_t<elem_bytes == 4, uint32_t, uint16_t>;
            uint32_t col_tile = block_first_tw;
            for (uint32_t p = 0; p < pages_this_block; ++p) {
                const uint32_t page = block_base + p * page_bytes;
                const uint32_t right_page = right_base + p * page_bytes;
                const uint32_t cols = logical_cols(col_tile);
                // Ascending j is safe in place: source column j + c >= j is read before it is overwritten.
                for (uint32_t j = 0; j < cols; ++j) {
                    const uint32_t src_col = j + col_offset_in_tile;
                    if (src_col < TILE_COLS) {
                        dataflow_kernel_lib::copy_tile_column<cb_slice_pages, Elem>(page, src_col, page, j);
                    } else {
                        dataflow_kernel_lib::copy_tile_column<cb_slice_pages, Elem>(
                            right_page, src_col - TILE_COLS, page, j);
                    }
                }
                col_tile = col_tile + 1 == tensor_out_wt_tiles ? 0 : col_tile + 1;
            }
        }
        cb_push_back(cb_slice_pages, block_pages);
        remaining -= pages_this_block;
    }
}
