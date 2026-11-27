// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_slice reader (NCRISC / NoC0): fetch_block (+ realign_columns_block in col_realign).
//
// For each block of `block_pages` output pages: reserve one block of cb_slice_pages, issue the
// reads of every page of the block back to back, one barrier, push. Rows land at their natural
// in-tile positions (see g6_slice_runs.hpp); the writer applies the row rotation.
//
// fetch_block is raw NoC: no kernel_lib dataflow helper gathers runs of byte offsets from a
// TensorAccessor page sequence into a CB block (op_design.md, "Helpers considered and rejected").

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp"
#include "g6_slice_runs.hpp"

void kernel_main() {
    constexpr uint32_t cb_slice_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_right_col_tile = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t block_pages = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_in_ht_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_in_wt_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in_lead1_extent = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_out_ht_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t tensor_out_wt_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t out_lead1_extent = get_compile_time_arg_val(9);
    constexpr uint32_t rows_last_tile_row = get_compile_time_arg_val(10);
    constexpr uint32_t cols_last_tile_col = get_compile_time_arg_val(11);
    constexpr bool ROW_RUNS = get_compile_time_arg_val(12) != 0;
    constexpr bool W_SHIFT = get_compile_time_arg_val(13) != 0;
    constexpr auto x_args = TensorAccessorArgs<14>();

    using Geometry = g6_slice::TileGeometry<page_bytes>;
    using Walk = g6_slice::OutPageWalk<
        tensor_in_ht_tiles,
        tensor_in_wt_tiles,
        in_lead1_extent,
        tensor_out_ht_tiles,
        tensor_out_wt_tiles,
        out_lead1_extent>;

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t lead_begin0 = get_common_arg_val<uint32_t>(1);
    const uint32_t lead_begin1 = get_common_arg_val<uint32_t>(2);
    const uint32_t first_input_tile_row = get_common_arg_val<uint32_t>(3);
    const uint32_t row_offset_in_tile = get_common_arg_val<uint32_t>(4);
    const uint32_t first_input_tile_col = get_common_arg_val<uint32_t>(5);
    const uint32_t col_offset_in_tile = get_common_arg_val<uint32_t>(6);

    const uint32_t core_pages = get_arg_val<uint32_t>(0);
    const uint32_t o0 = get_arg_val<uint32_t>(1);

    Walk walk;
    walk.o1 = get_arg_val<uint32_t>(2);
    walk.th = get_arg_val<uint32_t>(3);
    walk.tw = get_arg_val<uint32_t>(4);
    walk.in_page = (((lead_begin0 + o0) * in_lead1_extent + lead_begin1 + walk.o1) * tensor_in_ht_tiles +
                    first_input_tile_row + walk.th) *
                       tensor_in_wt_tiles +
                   first_input_tile_col + walk.tw;

    const auto x = TensorAccessor(x_args, x_addr, page_bytes);

    // Row classes: every tile row but an image's last has 32 output rows.
    [[maybe_unused]] g6_slice::RunList fetch_full, fetch_last, right_full, right_last;
    [[maybe_unused]] bool last_col_needs_right = false;
    if constexpr (ROW_RUNS) {
        // col_realign: output columns [0, 32 - c) come from the left input tile's columns [c, 32), the
        // rest from the right tile's columns [0, c); only the face halves holding those columns move.
        const uint32_t c = col_offset_in_tile;
        const uint32_t left_halves =
            W_SHIFT && c >= g6_slice::FACE_DIM ? g6_slice::COL_HALF_RIGHT : g6_slice::COL_HALF_BOTH;
        fetch_full = Geometry::fetch_runs(row_offset_in_tile, g6_slice::TILE_DIM, left_halves);
        fetch_last = Geometry::fetch_runs(row_offset_in_tile, rows_last_tile_row, left_halves);
        if constexpr (W_SHIFT) {
            const uint32_t right_halves = c <= g6_slice::FACE_DIM ? g6_slice::COL_HALF_LEFT : g6_slice::COL_HALF_BOTH;
            right_full = Geometry::fetch_runs(row_offset_in_tile, g6_slice::TILE_DIM, right_halves);
            right_last = Geometry::fetch_runs(row_offset_in_tile, rows_last_tile_row, right_halves);
            last_col_needs_right = cols_last_tile_col + c > g6_slice::TILE_DIM;
        }
    }

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    uint32_t pages_left = core_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // A ragged last block narrows only its NoC work; the handshake stays at block_pages.
        const uint32_t pages_this_block = pages_left < block_pages ? pages_left : block_pages;
        pages_left -= pages_this_block;

        {
            DeviceZoneScopedN("G6_CB_RESERVE");
            cb_reserve_back(cb_slice_pages, block_pages);
        }
        const uint32_t block_l1 = get_write_ptr(cb_slice_pages);
        [[maybe_unused]] const uint32_t right_l1 = W_SHIFT ? get_write_ptr(cb_right_col_tile) : 0;

        // fetch_block
        {
            DeviceZoneScopedN("G6_FETCH_ISSUE");
            uint32_t page_l1 = block_l1;
            [[maybe_unused]] uint32_t right_page_l1 = right_l1;
            for (uint32_t p = 0; p < pages_this_block; ++p) {
                if constexpr (ROW_RUNS) {
                    const g6_slice::RunList& runs = walk.last_tile_row() ? fetch_last : fetch_full;
                    for (uint32_t r = 0; r < runs.count; ++r) {
                        const g6_slice::ByteRun& run = runs.runs[r];
                        const uint32_t tile = walk.in_page + run.src_tile * tensor_in_wt_tiles;
                        noc_async_read<page_bytes>(
                            x.get_noc_addr(tile, run.src_offset), page_l1 + run.dst_offset, run.bytes);
                    }
                    if constexpr (W_SHIFT) {
                        if (!walk.last_tile_col() || last_col_needs_right) {
                            const g6_slice::RunList& right = walk.last_tile_row() ? right_last : right_full;
                            for (uint32_t r = 0; r < right.count; ++r) {
                                const g6_slice::ByteRun& run = right.runs[r];
                                const uint32_t tile = walk.in_page + 1 + run.src_tile * tensor_in_wt_tiles;
                                noc_async_read<page_bytes>(
                                    x.get_noc_addr(tile, run.src_offset), right_page_l1 + run.dst_offset, run.bytes);
                            }
                        }
                        right_page_l1 += page_bytes;
                    }
                } else {
                    noc_async_read<page_bytes>(x.get_noc_addr(walk.in_page), page_l1, page_bytes);
                }
                page_l1 += page_bytes;
                walk.advance();
            }
        }
        {
            DeviceZoneScopedN("G6_READ_BARRIER");
            noc_async_read_barrier();
        }

        // realign_columns_block: in place, ascending columns (column j + c >= j is read before it is
        // overwritten). Runs after the barrier, which invalidates the L1 cache for these loads.
        if constexpr (W_SHIFT) {
            DeviceZoneScopedN("G6_COL_REALIGN");
            static_assert(page_bytes == 2048 || page_bytes == 4096, "column realign needs 16- or 32-bit elements");
            using elem_t = std::conditional_t<page_bytes == 4096, uint32_t, uint16_t>;
            const uint32_t c = col_offset_in_tile;
            uint32_t page_l1 = block_l1;
            uint32_t right_page_l1 = right_l1;
            for (uint32_t p = 0; p < pages_this_block; ++p) {
                for (uint32_t j = 0; j < g6_slice::TILE_DIM; ++j) {
                    if (j + c < g6_slice::TILE_DIM) {
                        dataflow_kernel_lib::copy_tile_column<cb_slice_pages, elem_t>(page_l1, j + c, page_l1, j);
                    } else {
                        dataflow_kernel_lib::copy_tile_column<cb_slice_pages, elem_t>(
                            right_page_l1, j + c - g6_slice::TILE_DIM, page_l1, j);
                    }
                }
                page_l1 += page_bytes;
                right_page_l1 += page_bytes;
            }
        }

        cb_push_back(cb_slice_pages, block_pages);
    }
}
