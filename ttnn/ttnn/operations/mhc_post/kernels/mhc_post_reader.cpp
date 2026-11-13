// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (RISCV_1, NoC0).
//
// Per token-tile row t of this core:
//   load_coefficients_row: read the post / comb tile of row t (once when both are the same buffer) into the
//     private landing cb_coef_raw, then copy coefficient column k into column 0 of its own page of
//     cb_coef_cols (unpack-to-dest COL broadcast reads column 0 of faces 0 / 2). Page order:
//     [post_0..post_{n-1} | comb_00..comb_{n-1,n-1}] (comb row-major, comb[i][j] at page n + i*n + j).
//   load_block: per pass of block_cols units (column tiles c), one F tile and the n X tiles of stream i = 0..n-1
//     per unit, unit-major, one barrier per block; the ring holds IN_BLOCKS blocks.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t FACE_WORDS = 256;  // 16 x 16 fp32
constexpr uint32_t FACE_DIM = 16;

// Copy column `col` of a 32x32 fp32 tile into column 0 of `dst` (32 words: 16 in face 0, 16 in face 2).
inline void copy_column_to_col0(const volatile uint32_t* src, uint32_t col, volatile uint32_t* dst) {
    const uint32_t src_face_col = (col / FACE_DIM);
    const uint32_t c = col % FACE_DIM;
    for (uint32_t half = 0; half < 2; ++half) {
        const volatile uint32_t* s = src + (half * 2 + src_face_col) * FACE_WORDS + c;
        volatile uint32_t* d = dst + (half * 2) * FACE_WORDS;
        for (uint32_t r = 0; r < FACE_DIM; ++r) {
            d[r * FACE_DIM] = s[r * FACE_DIM];
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stream_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(2);
    constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t block_cols = get_compile_time_arg_val(6);
    constexpr uint32_t c_tiles = get_compile_time_arg_val(7);  // tensor_c_tiles
    constexpr uint32_t post_wt = get_compile_time_arg_val(8);
    constexpr uint32_t comb_wt = get_compile_time_arg_val(9);
    constexpr auto f_args = TensorAccessorArgs<10>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr uint32_t n_coef = n + n * n;

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t t_count = get_arg_val<uint32_t>(1);
    const uint32_t c_start = get_arg_val<uint32_t>(2);
    const uint32_t c_count = get_arg_val<uint32_t>(3);

    const uint32_t f_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t post_col = get_common_arg_val<uint32_t>(4);
    const uint32_t comb_col = get_common_arg_val<uint32_t>(5);

    const uint32_t f_page = get_tile_size(cb_sublayer_tiles);
    const uint32_t x_page = get_tile_size(cb_stream_tiles);
    const uint32_t coef_page = get_tile_size(cb_coef_raw);
    const auto f_acc = TensorAccessor(f_args, f_addr, f_page);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_page);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page);
    const bool coef_shared = (post_addr == comb_addr);  // packed mixes: one tile serves post and comb

    const uint32_t raw_post = get_write_ptr(cb_coef_raw);  // reader-private landing (never pushed)
    const uint32_t raw_comb = coef_shared ? raw_post : raw_post + coef_page;
    constexpr uint32_t coef_words = 1024;  // fp32 words per coefficient page

    for (uint32_t rb = 0; rb < t_count; rb += block_rows) {
        const uint32_t rows_in_block = (t_count - rb) < block_rows ? (t_count - rb) : block_rows;
        for (uint32_t r = 0; r < rows_in_block; ++r) {
            const uint32_t t = t_start + rb + r;

            // ---- load_coefficients_row ----
            noc_async_read(post_acc.get_noc_addr(t * post_wt), raw_post, coef_page);
            if (!coef_shared) {
                noc_async_read(comb_acc.get_noc_addr(t * comb_wt), raw_comb, coef_page);
            }
            noc_async_read_barrier();
            cb_reserve_back(cb_coef_cols, n_coef);
            volatile uint32_t* cols = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_coef_cols));
            const volatile uint32_t* post_src = reinterpret_cast<const volatile uint32_t*>(raw_post);
            const volatile uint32_t* comb_src = reinterpret_cast<const volatile uint32_t*>(raw_comb);
            for (uint32_t k = 0; k < n; ++k) {
                copy_column_to_col0(post_src, post_col + k, cols + k * coef_words);
            }
            for (uint32_t k = 0; k < n * n; ++k) {
                copy_column_to_col0(comb_src, comb_col + k, cols + (n + k) * coef_words);
            }
            cb_push_back(cb_coef_cols, n_coef);

            // ---- load_block, one pass = one block of block_cols ring slots, one barrier ----
            // A ragged pass reads only its cols_in_pass units but still occupies block_cols slots (filler,
            // never read): every consumer window then starts on a block boundary and never wraps the ring.
            const uint32_t f_row = t * c_tiles;
            const uint32_t x_row = t * n * c_tiles;
            for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
                const uint32_t cols_in_pass = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
                cb_reserve_back(cb_sublayer_tiles, block_cols);
                cb_reserve_back(cb_stream_tiles, block_cols * n);
                uint32_t f_l1 = get_write_ptr(cb_sublayer_tiles);
                uint32_t x_l1 = get_write_ptr(cb_stream_tiles);
                for (uint32_t u = 0; u < cols_in_pass; ++u) {
                    const uint32_t c = c_start + cb0 + u;
                    noc_async_read(f_acc.get_noc_addr(f_row + c), f_l1, f_page);
                    f_l1 += f_page;
                    for (uint32_t i = 0; i < n; ++i) {
                        noc_async_read(x_acc.get_noc_addr(x_row + i * c_tiles + c), x_l1, x_page);
                        x_l1 += x_page;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_sublayer_tiles, block_cols);
                cb_push_back(cb_stream_tiles, block_cols * n);
            }
        }
    }
}
