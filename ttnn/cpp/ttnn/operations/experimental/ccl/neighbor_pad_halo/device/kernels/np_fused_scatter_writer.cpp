// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Fused scatter phase for neighbor_pad_halo padded-output mode. Runs on cores DISJOINT from the fabric
// exchange (columns x>=1), concurrently with the exchange. Writes the padded output [outer, Hp, Wp, C]:
//   INTERIOR (t,h,w) -> padded[t, h+pH, w+pW]   from interior_src (input); NO dependency on the exchange
//     -> processed first, overlapping the fabric exchange.
//   BORDER  -> from the compact halo buffer [H-top|H-bot|W-left|W-right]; depends on the exchange, so
//     before its first border stick a core waits exchange_done >= num_readers (all fabric halo landed).
// Stick order is interior then 4 border sections (H-top|H-bot|W-left|W-right), t-major — see the src_noc
// and dst_page maps below. border_only: interior already present in padded_output, so work starts at the
// border (interior skipped, no overlap).
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);        // interior source (input)
    const uint32_t compact_addr = get_arg_val<uint32_t>(1);  // border source (compact halo buffer)
    const uint32_t dst_addr = get_arg_val<uint32_t>(2);      // padded output
    const uint32_t stick_start = get_arg_val<uint32_t>(3);
    const uint32_t stick_count = get_arg_val<uint32_t>(4);
    const uint32_t compact_ready_addr = get_arg_val<uint32_t>(5);  // local sem: W-readers signal compact ready
    const uint32_t num_readers = get_arg_val<uint32_t>(6);         // W-reader broadcast count (0 = interior-only)
    const uint32_t logical_h = get_arg_val<uint32_t>(7);           // 0 = no H masking
    const uint32_t device_h_offset = get_arg_val<uint32_t>(8);     // global H index of this shard's row 0
    const uint32_t logical_w = get_arg_val<uint32_t>(9);           // 0 = no W masking
    const uint32_t device_w_offset = get_arg_val<uint32_t>(10);    // global W index of this shard's col 0

    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t outer = get_compile_time_arg_val(1);  // B*T frames
    constexpr uint32_t Hp = get_compile_time_arg_val(2);     // padded H (Hd + 2*pH)
    constexpr uint32_t Wp = get_compile_time_arg_val(3);     // padded W (Wd + 2*pW)
    constexpr uint32_t Hd = get_compile_time_arg_val(4);     // interior H (H_dev)
    constexpr uint32_t Wd = get_compile_time_arg_val(5);     // interior W (W_dev)
    constexpr uint32_t pH = get_compile_time_arg_val(6);     // H halo per side
    constexpr uint32_t pW = get_compile_time_arg_val(7);     // W halo per side
    constexpr uint32_t cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t batch = get_compile_time_arg_val(9);  // sticks per read/write NOC-barrier group
    constexpr uint32_t border_only = get_compile_time_arg_val(10);

    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto compact_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<compact_args.next_compile_time_args_offset()>();
    const auto x_src = TensorAccessor(x_args, x_addr, page_size);
    const auto compact = TensorAccessor(compact_args, compact_addr, page_size);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_size);

    // Flat stick index gi runs [interior | H-top | H-bot | W-left | W-right]. These are the per-section
    // stick counts and their cumulative boundaries, so src_noc/dst_page can dispatch by `gi < boundary`.
    constexpr uint32_t frame_stride = Hp * Wp;   // padded sticks per frame
    constexpr uint32_t n_int = outer * Hd * Wd;  // interior stick count (end of interior section)
    constexpr uint32_t h_sec = outer * pH * Wd;  // sticks in one H-halo section (top or bottom)
    constexpr uint32_t w_sec = outer * Hp * pW;  // sticks in one W-halo section (left or right)
    constexpr uint32_t bb1 = n_int + h_sec;      // end of H-top   -> H-bot starts here
    constexpr uint32_t bb2 = bb1 + h_sec;        // end of H-bot   -> W-left starts here
    constexpr uint32_t bb3 = bb2 + w_sec;        // end of W-left  -> W-right starts here

    const uint32_t l1_base = get_write_ptr(cb_id);
    const uint32_t end = stick_start + stick_count;

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

    // Border sticks (gi >= n_int) depend on the fabric exchange, signalled by the W-readers incrementing
    // this core's local compact_ready sem (num_readers = W-reader count). Wait exactly once, before the
    // first border stick this core touches; interior sticks (gi < n_int) run first with no wait (overlap).
    // num_readers == 0 => interior-only injection (border filled by a separate op): never wait, never reset.
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(compact_ready_addr);
    bool waited = false;
    if (border_only && num_readers > 0) {
        noc_semaphore_wait_min(ready, num_readers);
        waited = true;
    }

    const bool do_mask = (logical_h > 0) || (logical_w > 0);
    // Logical mask for an interior stick: zeroed when its GLOBAL content (h,w) reaches logical_h/logical_w.
    auto interior_masked = [&](uint32_t gi) -> bool {
        const uint32_t t = gi / (Hd * Wd);
        const uint32_t rem = gi - t * (Hd * Wd);
        const uint32_t h = rem / Wd;
        const uint32_t w = rem - h * Wd;
        return (logical_h > 0 && device_h_offset + h >= logical_h) ||
               (logical_w > 0 && device_w_offset + w >= logical_w);
    };
    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
    auto zero_l1 = [&](uint32_t l1_addr) {
        uint32_t off = 0;
        for (; off + MEM_ZEROS_SIZE <= page_size; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros_noc, l1_addr + off, MEM_ZEROS_SIZE);
        }
        if (off < page_size) {
            noc_async_read(zeros_noc, l1_addr + off, page_size - off);
        }
    };

    for (uint32_t gi = stick_start; gi < end; gi += batch) {
        const uint32_t n = (end - gi < batch) ? (end - gi) : batch;
        if (!waited && num_readers > 0 && (gi + n) > n_int) {
            noc_semaphore_wait_min(ready, num_readers);
            waited = true;
        }
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t gk = gi + k;
            if (do_mask && gk < n_int && interior_masked(gk)) {
                zero_l1(l1_base + k * page_size);
            } else {
                noc_async_read(src_noc(gk), l1_base + k * page_size, page_size);
            }
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < n; k++) {
            noc_async_write(l1_base + k * page_size, get_noc_addr(dst_page(gi + k), dst), page_size);
        }
        noc_async_write_barrier();
    }

    // Trace-safe self-reset: this core received exactly num_readers increments; an interior-only core
    // (no border in its range) never waited above, so wait now to guarantee all increments have landed
    // before zeroing — otherwise a late increment would leave the sem non-zero for the next dispatch.
    if (num_readers > 0) {
        if (!waited) {
            noc_semaphore_wait_min(ready, num_readers);
        }
        noc_semaphore_set(ready, 0);
    }
}
