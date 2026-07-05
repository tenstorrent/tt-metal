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
// Compact stick order is section-major then (t, row, col) t-major, matching the halo op's output.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);

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

    const uint32_t frame_stride = Hp * Wp;
    const uint32_t l1_base = get_write_ptr(cb_id);
    uint32_t sp = 0;    // source (compact) page index, in reference order
    uint32_t ring = 0;  // rotating L1 scratch page to overlap successive read/write pairs

    // Copy one compact stick sp -> padded page dpage.
    auto copy = [&](uint32_t dpage) {
        const uint32_t l1 = l1_base + ring * page_size;
        noc_async_read(get_noc_addr(sp, src), l1, page_size);
        noc_async_read_barrier();
        noc_async_write(l1, get_noc_addr(dpage, dst), page_size);
        noc_async_write_barrier();
        sp++;
        ring = (ring + 1 == cb_pages) ? 0 : ring + 1;
    };

    // H-top: rows [0, pH), interior cols [pW, pW+Wd)
    for (uint32_t t = 0; t < outer; t++) {
        const uint32_t fb = t * frame_stride;
        for (uint32_t pr = 0; pr < pH; pr++) {
            const uint32_t rb = fb + pr * Wp + pW;
            for (uint32_t w = 0; w < Wd; w++) {
                copy(rb + w);
            }
        }
    }
    // H-bot: rows [pH+Hd, pH+Hd+pH), interior cols [pW, pW+Wd)
    for (uint32_t t = 0; t < outer; t++) {
        const uint32_t fb = t * frame_stride;
        for (uint32_t pr = 0; pr < pH; pr++) {
            const uint32_t rb = fb + (pH + Hd + pr) * Wp + pW;
            for (uint32_t w = 0; w < Wd; w++) {
                copy(rb + w);
            }
        }
    }
    // W-left: all rows [0, Hp) (incl. corners), cols [0, pW)
    for (uint32_t t = 0; t < outer; t++) {
        const uint32_t fb = t * frame_stride;
        for (uint32_t hp = 0; hp < Hp; hp++) {
            const uint32_t rb = fb + hp * Wp;
            for (uint32_t wc = 0; wc < pW; wc++) {
                copy(rb + wc);
            }
        }
    }
    // W-right: all rows [0, Hp) (incl. corners), cols [pW+Wd, pW+Wd+pW)
    for (uint32_t t = 0; t < outer; t++) {
        const uint32_t fb = t * frame_stride;
        for (uint32_t hp = 0; hp < Hp; hp++) {
            const uint32_t rb = fb + hp * Wp + (pW + Wd);
            for (uint32_t wc = 0; wc < pW; wc++) {
                copy(rb + wc);
            }
        }
    }
}
