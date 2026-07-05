// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Local border scatter: copy the compact halo buffer [H-top | H-bot | W-left | W-right] into the
// BORDER of a persistent padded buffer [outer, Hp, Wp, C] in place. Each compact stick maps to a
// fixed padded page — the exact inverse of compact_halo_reference() in the NP-halo test:
//   H-top  stick (t, pr, w)  -> padded[t, pr,          pW + w]         (interior W cols)
//   H-bot  stick (t, pr, w)  -> padded[t, pH + Hd + pr, pW + w]
//   W-left stick (t, hp, wc) -> padded[t, hp,           wc]            (full height, incl corners)
//   W-right stick(t, hp, wc) -> padded[t, hp,           pW + Wd + wc]
// Compact stick order is section-major then (t, row, col) t-major. This kernel is parallelized by
// GLOBAL stick index: each core handles a contiguous [start, start+count) range and maps each global
// index to its section + (t,row,col) -> padded page. The compact source page IS the global index.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t stick_start = get_arg_val<uint32_t>(2);
    const uint32_t stick_count = get_arg_val<uint32_t>(3);

    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t outer = get_compile_time_arg_val(1);  // B*T frames
    constexpr uint32_t Hp = get_compile_time_arg_val(2);     // padded H = Hd + 2*pH
    constexpr uint32_t Wp = get_compile_time_arg_val(3);     // padded W = Wd + 2*pW
    constexpr uint32_t Hd = get_compile_time_arg_val(4);
    constexpr uint32_t Wd = get_compile_time_arg_val(5);
    constexpr uint32_t pH = get_compile_time_arg_val(6);
    constexpr uint32_t pW = get_compile_time_arg_val(7);
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_pages = get_compile_time_arg_val(9);

    constexpr auto src_args = TensorAccessorArgs<10>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr, page_size);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_size);

    // Section sizes (sticks) and cumulative boundaries in the compact/global order.
    constexpr uint32_t h_sec = outer * pH * Wd;   // H-top == H-bot
    constexpr uint32_t w_sec = outer * Hp * pW;   // W-left == W-right
    constexpr uint32_t b1 = h_sec;                // end of H-top
    constexpr uint32_t b2 = h_sec + h_sec;        // end of H-bot
    constexpr uint32_t b3 = b2 + w_sec;           // end of W-left
    constexpr uint32_t frame_stride = Hp * Wp;

    const uint32_t l1_base = get_write_ptr(cb_id);
    uint32_t ring = 0;
    const uint32_t end = stick_start + stick_count;
    for (uint32_t gi = stick_start; gi < end; gi++) {
        uint32_t dpage;
        if (gi < b1) {
            // H-top: rows [0, pH), interior cols [pW, pW+Wd)
            const uint32_t j = gi;
            const uint32_t t = j / (pH * Wd);
            const uint32_t rem = j - t * (pH * Wd);
            const uint32_t pr = rem / Wd;
            const uint32_t w = rem - pr * Wd;
            dpage = t * frame_stride + pr * Wp + (pW + w);
        } else if (gi < b2) {
            // H-bot: rows [pH+Hd, pH+Hd+pH), interior cols [pW, pW+Wd)
            const uint32_t j = gi - b1;
            const uint32_t t = j / (pH * Wd);
            const uint32_t rem = j - t * (pH * Wd);
            const uint32_t pr = rem / Wd;
            const uint32_t w = rem - pr * Wd;
            dpage = t * frame_stride + (pH + Hd + pr) * Wp + (pW + w);
        } else if (gi < b3) {
            // W-left: all rows [0, Hp), cols [0, pW)
            const uint32_t j = gi - b2;
            const uint32_t t = j / (Hp * pW);
            const uint32_t rem = j - t * (Hp * pW);
            const uint32_t hp = rem / pW;
            const uint32_t wc = rem - hp * pW;
            dpage = t * frame_stride + hp * Wp + wc;
        } else {
            // W-right: all rows [0, Hp), cols [pW+Wd, pW+Wd+pW)
            const uint32_t j = gi - b3;
            const uint32_t t = j / (Hp * pW);
            const uint32_t rem = j - t * (Hp * pW);
            const uint32_t hp = rem / pW;
            const uint32_t wc = rem - hp * pW;
            dpage = t * frame_stride + hp * Wp + (pW + Wd + wc);
        }
        const uint32_t l1 = l1_base + ring * page_size;
        noc_async_read(get_noc_addr(gi, src), l1, page_size);
        noc_async_read_barrier();
        noc_async_write(l1, get_noc_addr(dpage, dst), page_size);
        noc_async_write_barrier();
        ring = (ring + 1 == cb_pages) ? 0 : ring + 1;
    }
}
