// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (NCRISC, NoC0).
//
// Per segment (one token-tile row of this core's unit range):
//   load_coefficients — read the raw post / comb tiles of row r into cb_coef_raw (one barrier), then
//                       expand them into n + n^2 column-broadcast fp32 tiles in cb_coef_bcast:
//                         expanded[k](rho, gamma) = post(rho, k)          for k <  n
//                         expanded[n + m](rho, gamma) = comb(rho, m)       for m <  n*n  (m = i*n + j)
//                       The expansion is scheduled inside the read-barrier shadow of the segment's first
//                       data block (block reads issued, expansion done, then the barrier).
//   load_block        — per block of block_col_tiles columns: B F tiles (slot c) and n*B X tiles
//                       (slot i*B + c), valid columns only, ONE barrier, nominal pushes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "mhc_post_common.hpp"

namespace {

constexpr uint32_t FACE_HW = 16;
constexpr uint32_t FACE_ELEMS = FACE_HW * FACE_HW;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t FACE_ROW_BYTES = FACE_HW * FP32_BYTES;

// Byte offset of element (row, col) inside a 32x32 fp32 tile (face-major layout).
FORCE_INLINE uint32_t fp32_tile_elem_offset(uint32_t row, uint32_t col) {
    const uint32_t face = ((row >= FACE_HW) ? 2u : 0u) + ((col >= FACE_HW) ? 1u : 0u);
    return (face * FACE_ELEMS + (row % FACE_HW) * FACE_HW + (col % FACE_HW)) * FP32_BYTES;
}

// Column-broadcast expansion: every element (rho, gamma) of the tile at dst_tile_addr becomes raw(rho, col).
FORCE_INLINE void expand_column(uint32_t raw_tile_addr, uint32_t col, uint32_t dst_tile_addr, uint32_t tile_rows) {
    for (uint32_t rho = 0; rho < tile_rows; ++rho) {
        const uint32_t bits =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_tile_addr + fp32_tile_elem_offset(rho, col));
        // Left half (face 0 / 2) and right half (face 1 / 3) of row rho.
        const uint32_t left = dst_tile_addr + fp32_tile_elem_offset(rho, 0);
        const uint32_t right = dst_tile_addr + fp32_tile_elem_offset(rho, FACE_HW);
        dataflow_kernel_lib::fill_l1_range<FP32_BYTES>(left, FACE_ROW_BYTES, bits);
        dataflow_kernel_lib::fill_l1_range<FP32_BYTES>(right, FACE_ROW_BYTES, bits);
    }
}

}  // namespace

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t n = get_compile_time_arg_val(0);                   // streams
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);   // Ct = C / 32
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);     // B
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(3);  // ceil(n / 32)
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(4);  // ceil(n*n / 32)
    constexpr uint32_t sublayer_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t residual_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t coef_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(8);  // 32
    constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(11);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(12);
    constexpr auto sublayer_args = TensorAccessorArgs<13>();
    constexpr auto residual_args = TensorAccessorArgs<sublayer_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_coef_tiles = n + n * n;
    constexpr uint32_t num_raw_tiles = post_tiles_per_row + comb_tiles_per_row;
    constexpr uint32_t residual_row_tiles = n * col_tiles_per_row;
    constexpr uint32_t residual_block_tiles = n * block_col_tiles;

    // ---- runtime args ----
    const uint32_t sublayer_addr = get_arg_val<uint32_t>(0);
    const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    const auto sublayer_acc = TensorAccessor(sublayer_args, sublayer_addr, sublayer_page_bytes);
    const auto residual_acc = TensorAccessor(residual_args, residual_addr, residual_page_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_page_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page_bytes);

    // Issue the async reads of one data block (no barrier) into already-reserved CB windows.
    auto issue_block_reads = [&](uint32_t row, uint32_t col_start, uint32_t valid_cols) {
        const uint32_t f_base = get_write_ptr(cb_sublayer_tiles);
        const uint32_t x_base = get_write_ptr(cb_residual_tiles);
        const uint32_t f_page0 = row * col_tiles_per_row + col_start;
        for (uint32_t c = 0; c < valid_cols; ++c) {
            noc_async_read(
                sublayer_acc.get_noc_addr(f_page0 + c), f_base + c * sublayer_page_bytes, sublayer_page_bytes);
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x_page0 = row * residual_row_tiles + i * col_tiles_per_row + col_start;
            const uint32_t x_slot0 = x_base + i * block_col_tiles * residual_page_bytes;
            for (uint32_t c = 0; c < valid_cols; ++c) {
                noc_async_read(
                    residual_acc.get_noc_addr(x_page0 + c), x_slot0 + c * residual_page_bytes, residual_page_bytes);
            }
        }
    };

    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
    while (!walker.done()) {
        const mhc_post::Segment seg = walker.next();
        const uint32_t blocks = mhc_post::num_blocks(seg.col_tiles, block_col_tiles);

        // ---- load_coefficients, part 1: raw post / comb tiles of this token row ----
        cb_reserve_back(cb_coef_raw, num_raw_tiles);
        const uint32_t raw_post_addr = get_write_ptr(cb_coef_raw);
        const uint32_t raw_comb_addr = raw_post_addr + post_tiles_per_row * coef_page_bytes;
        noc_async_read(post_acc.get_noc_addr(seg.row * post_tiles_per_row), raw_post_addr, coef_page_bytes);
        noc_async_read(comb_acc.get_noc_addr(seg.row * comb_tiles_per_row), raw_comb_addr, coef_page_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_coef_raw, num_raw_tiles);
        cb_wait_front(cb_coef_raw, num_raw_tiles);

        for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
            const uint32_t valid = mhc_post::block_valid_col_tiles(seg.col_tiles, block_col_tiles, block_idx);
            const uint32_t col_start = seg.col0 + block_idx * block_col_tiles;

            // ---- load_block: issue reads ----
            cb_reserve_back(cb_sublayer_tiles, block_col_tiles);
            cb_reserve_back(cb_residual_tiles, residual_block_tiles);
            issue_block_reads(seg.row, col_start, valid);

            if (block_idx == 0) {
                // ---- load_coefficients, part 2: expansion in the first block's read-barrier shadow ----
                cb_reserve_back(cb_coef_bcast, num_coef_tiles);
                const uint32_t bcast_base = get_write_ptr(cb_coef_bcast);
                for (uint32_t k = 0; k < n; ++k) {
                    expand_column(raw_post_addr, k, bcast_base + k * coef_page_bytes, tile_rows);
                }
                for (uint32_t m = 0; m < n * n; ++m) {
                    expand_column(raw_comb_addr, m, bcast_base + (n + m) * coef_page_bytes, tile_rows);
                }
                cb_push_back(cb_coef_bcast, num_coef_tiles);
                cb_pop_front(cb_coef_raw, num_raw_tiles);
            }

            noc_async_read_barrier();
            cb_push_back(cb_sublayer_tiles, block_col_tiles);
            cb_push_back(cb_residual_tiles, residual_block_tiles);
        }
    }
}
