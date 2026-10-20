// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per position p = r * Ct + c (token-tile r, column-tile c): streams the n stream tiles x[r, i, c]
// and the tile y[r, c]. Per token-tile r (once, before its first position): fetches the comb and
// post tiles of r and builds the n*n + n coefficient COLUMN tiles -- tile k holds coefficient k
// (comb columns i*n + j, then the n post columns) in column 0 of every row, zeros elsewhere -- for
// the compute kernel's 32-bit column-broadcast datacopy.
//
// fp32 tile layout (32x32, 4 faces of 16x16, row-major within a face): datum (t, c) sits at word
// ((t / 16) * 2 + c / 16) * 256 + (t % 16) * 16 + c % 16.
namespace {
FORCE_INLINE uint32_t tile_word(uint32_t t, uint32_t c) {
    return ((t >> 4) * 2 + (c >> 4)) * 256 + (t & 15) * 16 + (c & 15);
}
}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_pos = get_arg_val<uint32_t>(4);
    const uint32_t num_pos = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_raw = get_compile_time_arg_val(2);
    constexpr uint32_t cb_col = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t Ct = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    constexpr uint32_t n_coef = n * n + n;

    const uint32_t x_page = get_local_cb_interface(cb_x).fifo_page_size;
    const uint32_t y_page = get_local_cb_interface(cb_y).fifo_page_size;
    const uint32_t raw_page = get_local_cb_interface(cb_raw).fifo_page_size;
    const uint32_t col_page = get_local_cb_interface(cb_col).fifo_page_size;

    const auto s_x = TensorAccessor(x_args, x_addr);
    const auto s_y = TensorAccessor(y_args, y_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);

    Noc noc;
    CircularBuffer cbx(cb_x), cby(cb_y), cbraw(cb_raw), cbcol(cb_col);

    // The raw-tile scratch is reserved once and never pushed: it is private to this kernel.
    cbraw.reserve_back(2);
    const uint32_t raw_base = get_write_ptr(cb_raw);
    volatile tt_l1_ptr uint32_t* comb_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_base);
    volatile tt_l1_ptr uint32_t* post_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_base + raw_page);

    uint32_t cur_r = 0xFFFFFFFFu;
    bool col_zeroed = false;
    for (uint32_t p = start_pos; p < start_pos + num_pos; ++p) {
        const uint32_t r = p / Ct;
        const uint32_t c = p - r * Ct;

        if (r != cur_r) {
            // comb and post tiles of token-tile r
            noc.async_read(s_comb, cbraw, raw_page, {.page_id = r}, {.offset_bytes = 0});
            noc.async_read(s_post, cbraw, raw_page, {.page_id = r}, {.offset_bytes = raw_page});
            noc.async_read_barrier();

            // the coefficient column tiles (the CB holds exactly n_coef tiles, so this reserve waits
            // for the compute kernel to have consumed the previous token-tile's set)
            cbcol.reserve_back(n_coef);
            const uint32_t col_base = get_write_ptr(cb_col);
            if (!col_zeroed) {
                // zero the whole set once per core (from the CB's write pointer); only column 0 is
                // rewritten per token-tile
                noc.async_write_zeros(cbcol, n_coef * col_page);
                noc.write_zeros_l1_barrier();
                col_zeroed = true;
            }
            for (uint32_t k = 0; k < n_coef; ++k) {
                volatile tt_l1_ptr uint32_t* src = (k < n * n) ? comb_w : post_w;
                const uint32_t kc = (k < n * n) ? k : k - n * n;
                volatile tt_l1_ptr uint32_t* dst =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(col_base + k * col_page);
                for (uint32_t t = 0; t < 32; ++t) {
                    dst[tile_word(t, 0)] = src[tile_word(t, kc)];
                }
            }
            cbcol.push_back(n_coef);
            cur_r = r;
        }

        cbx.reserve_back(n);
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_read(s_x, cbx, x_page, {.page_id = (r * n + i) * Ct + c}, {.offset_bytes = i * x_page});
        }
        cby.reserve_back(1);
        noc.async_read(s_y, cby, y_page, {.page_id = r * Ct + c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbx.push_back(n);
        cby.push_back(1);
    }
}
