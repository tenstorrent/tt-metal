// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle, channels_last, COLUMN_MAJOR_PADDED: reader. Walks this core's output tiles (32/r pixels
// each) image row by image row; for every run of tiles inside one image row it reads the C*r input rows
// (row k = dy*C + c of the block) for those columns into one [32 x W] page of cb_in. Rows C*r..31 of both
// pages are zeroed once at start and never written again.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t r = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t cb_in = get_compile_time_arg_val(5);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7, 0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile0 = get_arg_val<uint32_t>(1);
    const uint32_t ntiles = get_arg_val<uint32_t>(2);
    if (ntiles == 0) {
        return;
    }
    constexpr uint32_t CR = C * r;
    constexpr uint32_t Ho = H / r;
    constexpr uint32_t page_bytes = 32 * row_bytes;
    constexpr uint32_t tile_col_bytes = 32 * 2;

    const auto s_in = TensorAccessor(src_args, src_addr);

    // zero the pad rows of both pages (the write pointer is at the CB base: page p is at base + p*page_bytes)
    if constexpr (CR < 32) {
        uint32_t dst = get_write_ptr(cb_in) + CR * row_bytes;
        for (uint32_t p = 0; p < 2; p++) {
            uint32_t n = (32 - CR) * row_bytes;
            uint32_t a = dst;
            while (n) {
                const uint32_t sz = n < MEM_ZEROS_SIZE ? n : MEM_ZEROS_SIZE;
                noc_async_write(static_cast<uint32_t>(MEM_ZEROS_BASE), get_noc_addr(a), sz);
                a += sz;
                n -= sz;
            }
            dst += page_bytes;
        }
        noc_async_write_barrier();
    }

    uint32_t g = tile0;
    const uint32_t end = tile0 + ntiles;
    while (g < end) {
        const uint32_t row = g / Wt;  // image row index n*Ho + oh
        const uint32_t tr = g - row * Wt;
        uint32_t run = Wt - tr;
        if (run > end - g) {
            run = end - g;
        }
        const uint32_t n = row / Ho;
        const uint32_t oh = row - n * Ho;
        const uint32_t page_base = n * C * H + oh * r;  // + c*H + dy
        cb_reserve_back(cb_in, 1);
        uint32_t dst = get_write_ptr(cb_in);
        const uint32_t src_off = tr * tile_col_bytes;
        const uint32_t nbytes = run * tile_col_bytes;
        for (uint32_t dy = 0; dy < r; dy++) {
            for (uint32_t c = 0; c < C; c++) {
                noc_async_read(s_in.get_noc_addr(page_base + c * H + dy, src_off), dst, nbytes);
                dst += row_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
        g += run;
    }
}
