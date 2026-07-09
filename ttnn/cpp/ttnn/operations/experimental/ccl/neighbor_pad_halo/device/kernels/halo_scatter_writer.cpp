// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Repack into a persistent padded buffer [outer, Hp, Wp, C]: fill the INTERIOR from the unpadded
// activation `x` and the BORDER from the compact halo buffer [H-top | H-bot | W-left | W-right], in a
// single pass that writes every padded page exactly once (folds the old ttnn.pad + border-scatter).
//   interior stick (t, h, w)  -> padded[t, h+pH,       w+pW]         (src: x page t*Hd*Wd + h*Wd + w)
//   H-top  stick (t, pr, w)   -> padded[t, pr,          pW+w]        (src: compact)
//   H-bot  stick (t, pr, w)   -> padded[t, pH+Hd+pr,    pW+w]
//   W-left stick (t, hp, wc)  -> padded[t, hp,          wc]          (full height, incl corners)
//   W-right stick(t, hp, wc)  -> padded[t, hp,          pW+Wd+wc]
// Compact section order + (t,row,col) t-major matches compact_halo_reference(). Parallelized by a
// single global stick index: [0, N_int) are interior sticks (from x), [N_int, ...) are compact border
// sticks; each core owns a contiguous [start, start+count) range. Pages of interleaved DRAM are
// bank-distributed (no contiguous multi-page read), so throughput comes from BATCHING the async
// reads/writes (one barrier per batch of cb_pages) rather than a per-stick read+write+barrier.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);        // interior source (unpadded activation)
    const uint32_t compact_addr = get_arg_val<uint32_t>(1);  // border source (compact halo buffer)
    const uint32_t dst_addr = get_arg_val<uint32_t>(2);      // padded output
    const uint32_t stick_start = get_arg_val<uint32_t>(3);
    const uint32_t stick_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t outer = get_compile_time_arg_val(1);  // B*T frames
    constexpr uint32_t Hp = get_compile_time_arg_val(2);     // padded H = Hd + 2*pH
    constexpr uint32_t Wp = get_compile_time_arg_val(3);     // padded W = Wd + 2*pW
    constexpr uint32_t Hd = get_compile_time_arg_val(4);
    constexpr uint32_t Wd = get_compile_time_arg_val(5);
    constexpr uint32_t pH = get_compile_time_arg_val(6);
    constexpr uint32_t pW = get_compile_time_arg_val(7);
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t batch = get_compile_time_arg_val(9);  // sticks per barrier batch (= cb pages)

    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto compact_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<compact_args.next_compile_time_args_offset()>();
    const auto x_src = TensorAccessor(x_args, x_addr, page_size);
    const auto compact = TensorAccessor(compact_args, compact_addr, page_size);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_size);

    constexpr uint32_t frame_stride = Hp * Wp;
    constexpr uint32_t n_int = outer * Hd * Wd;  // interior sticks
    constexpr uint32_t h_sec = outer * pH * Wd;  // H-top == H-bot (compact)
    constexpr uint32_t w_sec = outer * Hp * pW;  // W-left == W-right (compact)
    constexpr uint32_t bb1 = n_int + h_sec;      // end of H-top (global index)
    constexpr uint32_t bb2 = bb1 + h_sec;        // end of H-bot
    constexpr uint32_t bb3 = bb2 + w_sec;        // end of W-left

    const uint32_t l1_base = get_write_ptr(cb_id);
    const uint32_t end = stick_start + stick_count;

    // Map a global stick index to (src noc addr, dst page).
    auto src_noc = [&](uint32_t gi) -> uint64_t {
        if (gi < n_int) {
            return get_noc_addr(gi, x_src);
        } else if (gi < bb1) {
            return get_noc_addr(gi - n_int, compact);
        } else if (gi < bb2) {
            return get_noc_addr(h_sec + (gi - bb1), compact);
        } else if (gi < bb3) {
            return get_noc_addr(2 * h_sec + (gi - bb2), compact);
        } else {
            return get_noc_addr(2 * h_sec + w_sec + (gi - bb3), compact);
        }
    };
    auto dst_page = [&](uint32_t gi) -> uint32_t {
        if (gi < n_int) {
            const uint32_t t = gi / (Hd * Wd);
            const uint32_t rem = gi - t * (Hd * Wd);
            const uint32_t h = rem / Wd;
            const uint32_t w = rem - h * Wd;
            return t * frame_stride + (h + pH) * Wp + (pW + w);
        } else if (gi < bb1) {
            const uint32_t j = gi - n_int;
            const uint32_t t = j / (pH * Wd);
            const uint32_t rem = j - t * (pH * Wd);
            const uint32_t pr = rem / Wd;
            const uint32_t w = rem - pr * Wd;
            return t * frame_stride + pr * Wp + (pW + w);
        } else if (gi < bb2) {
            const uint32_t j = gi - bb1;
            const uint32_t t = j / (pH * Wd);
            const uint32_t rem = j - t * (pH * Wd);
            const uint32_t pr = rem / Wd;
            const uint32_t w = rem - pr * Wd;
            return t * frame_stride + (pH + Hd + pr) * Wp + (pW + w);
        } else if (gi < bb3) {
            const uint32_t j = gi - bb2;
            const uint32_t t = j / (Hp * pW);
            const uint32_t rem = j - t * (Hp * pW);
            const uint32_t hp = rem / pW;
            const uint32_t wc = rem - hp * pW;
            return t * frame_stride + hp * Wp + wc;
        } else {
            const uint32_t j = gi - bb3;
            const uint32_t t = j / (Hp * pW);
            const uint32_t rem = j - t * (Hp * pW);
            const uint32_t hp = rem / pW;
            const uint32_t wc = rem - hp * pW;
            return t * frame_stride + hp * Wp + (pW + Wd + wc);
        }
    };

    for (uint32_t gi = stick_start; gi < end; gi += batch) {
        const uint32_t n = (end - gi < batch) ? (end - gi) : batch;
        // Batch of reads: src -> L1 ring slots, one barrier.
        for (uint32_t k = 0; k < n; k++) {
            noc_async_read(src_noc(gi + k), l1_base + k * page_size, page_size);
        }
        noc_async_read_barrier();
        // Batch of writes: L1 ring slots -> padded pages, one barrier.
        for (uint32_t k = 0; k < n; k++) {
            noc_async_write(l1_base + k * page_size, get_noc_addr(dst_page(gi + k), dst), page_size);
        }
        noc_async_write_barrier();
    }
}
