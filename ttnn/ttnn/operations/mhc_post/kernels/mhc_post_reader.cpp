// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (NCRISC, NoC0).
//
// Per segment (one token-tile row of this core's unit range):
//   load_coefficients — read the raw post / comb tiles of row r into cb_coef_raw (one barrier), then
//                       expand them into n * P half-packed column-broadcast fp32 tiles in cb_coef_bcast
//                       (layout: mhc_post_common.hpp — term t of stream j in half t%2 of tile j*P + t/2;
//                       term 0 = post(rho, j), term 1+i = comb(rho, i*n + j)).
//                       The expansion is scheduled inside the read-barrier shadow of the segment's first
//                       data block (block reads issued, expansion done, then the barrier).
//   load_block        — per block of block_col_tiles columns: B F tiles (slot c) and n*B X tiles
//                       (slot i*B + c), valid columns only, ONE barrier, nominal pushes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_common.hpp"

namespace {

constexpr uint32_t FACE_HW = 16;
constexpr uint32_t FACE_ELEMS = FACE_HW * FACE_HW;

// Half-tile column-broadcast expansion: every element (rho, gamma) of half `half` (gamma in
// [16*half, 16*half + 16)) of the fp32 tile at dst_tile_addr becomes raw(rho, col). See mhc_post_common.hpp.
// The half is two contiguous faces (2*fh + half, fh = row half), each 16 rows x 16 words; one raw load per
// row feeds 16 unrolled word stores (no per-element address math, no per-row helper call).
FORCE_INLINE void expand_half(uint32_t raw_tile_addr, uint32_t col, uint32_t dst_tile_addr, uint32_t half) {
#pragma GCC unroll 1
    for (uint32_t fh = 0; fh < 2; ++fh) {
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_tile_addr) +
                                                 (2 * fh) * FACE_ELEMS + (col % FACE_HW) + (col / FACE_HW) * FACE_ELEMS;
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile_addr) + (2 * fh + half) * FACE_ELEMS;
#pragma GCC unroll 1
        for (uint32_t r = 0; r < FACE_HW; ++r) {
            const uint32_t bits = src[r * FACE_HW];
#pragma GCC unroll 16
            for (uint32_t w = 0; w < FACE_HW; ++w) {
                dst[w] = bits;
            }
            dst += FACE_HW;
        }
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
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(13);  // P = ceil((n+1)/2)
    constexpr auto sublayer_args = TensorAccessorArgs<14>();
    constexpr auto residual_args = TensorAccessorArgs<sublayer_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    static_assert(tile_rows == 2 * FACE_HW, "mhc_post reader: expansion assumes 32x32 tiles of 16x16 faces");

    constexpr uint32_t num_coef_tiles = n * coef_tiles_per_stream;
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
#ifdef ABL_NO_DM
        return;
#endif
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

#ifdef ABL_NO_EXPAND
            if (false) {
#else
            if (block_idx == 0) {
#endif
                // ---- load_coefficients, part 2: expansion in the first block's read-barrier shadow ----
                cb_reserve_back(cb_coef_bcast, num_coef_tiles);
                const uint32_t bcast_base = get_write_ptr(cb_coef_bcast);
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t t = 0; t <= n; ++t) {
                        const uint32_t raw_addr = t == 0 ? raw_post_addr : raw_comb_addr;
                        const uint32_t raw_col = t == 0 ? j : (t - 1) * n + j;
                        const uint32_t tile = j * coef_tiles_per_stream + mhc_post::coef_tile_in_stream(t);
                        expand_half(raw_addr, raw_col, bcast_base + tile * coef_page_bytes, mhc_post::coef_half(t));
                    }
                }
            }

            noc_async_read_barrier();  // data block
            if (block_idx == 0) {
#ifdef ABL_NO_EXPAND
                cb_reserve_back(cb_coef_bcast, num_coef_tiles);
#endif
                cb_push_back(cb_coef_bcast, num_coef_tiles);
                cb_pop_front(cb_coef_raw, num_raw_tiles);
            }
            cb_push_back(cb_sublayer_tiles, block_col_tiles);
            cb_push_back(cb_residual_tiles, residual_block_tiles);
        }
    }
}
