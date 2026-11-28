// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// host_move_list "ideal floor": one kernel per core, the core's transfers are compile-time constants
// (bank mode encoding of hml_mover.cpp, starting at compile arg 5). Only the two buffer bases are runtime.
// Not cache-safe for the real op (one binary per core); it bounds what any device-side mover can reach.

#include "hml_common.hpp"

template <uint32_t I>
FORCE_INLINE constexpr uint32_t ct() {
    return get_compile_time_arg_val(I);
}

template <uint32_t kBase, uint32_t kN, bool kRead, uint32_t kI = 0>
FORCE_INLINE void xfers(uint32_t base, uint32_t l1, uint8_t noc) {
    if constexpr (kI < kN) {
        constexpr uint32_t w0 = ct<kBase + 3 * kI>();
        constexpr uint32_t w1 = ct<kBase + 3 * kI + 1>();
        constexpr uint32_t w2 = ct<kBase + 3 * kI + 2>();
        const uint64_t d = get_noc_addr_from_bank_id<true>(w0 & 0xFF, base + w1, noc);
        if constexpr (kRead) {
            noc_async_read(d, l1 + w2, w0 >> 8, noc);
        } else {
            noc_async_write(l1 + w2, d, w0 >> 8, noc);
        }
        xfers<kBase, kN, kRead, kI + 1>(base, l1, noc);
    }
}

template <uint32_t kElem, uint32_t kBase, uint32_t kN, uint32_t kI = 0>
FORCE_INLINE uint32_t copies(uint32_t stage, uint32_t image) {
    if constexpr (kI < kN) {
        constexpr uint32_t s = ct<kBase + 3 * kI>();
        constexpr uint32_t d = ct<kBase + 3 * kI + 1>();
        constexpr uint32_t n = ct<kBase + 3 * kI + 2>();
        hml::copy_run<kElem>(stage + s, image + d, n);
        const uint32_t rest = copies<kElem, kBase, kN, kI + 1>(stage, image);
        return kI + 1 < kN ? rest : image + d + n - 1;
    }
    return image;
}

void kernel_main() {
    constexpr uint32_t kElem = ct<0>();
    constexpr uint32_t cb_stage = ct<2>();
    constexpr uint32_t cb_image = ct<3>();
    constexpr uint32_t n_reads = ct<5>();
    constexpr uint32_t n_zc = ct<6>();
    constexpr uint32_t n_copies = ct<7>();
    constexpr uint32_t n_imgw = ct<8>();
    constexpr uint32_t kR = 9;
    constexpr uint32_t kZ = kR + 3 * n_reads;
    constexpr uint32_t kC = kZ + 3 * n_zc;
    constexpr uint32_t kW = kC + 3 * n_copies;
    if constexpr (n_reads > 0) {
        const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
        const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
        const uint8_t noc = noc_index;
        const uint32_t stage = (get_write_ptr(cb_stage) + 63) & ~63u;
        const uint32_t image = (get_write_ptr(cb_image) + 63) & ~63u;
        xfers<kR, n_reads, true>(in_addr, stage, noc);
        noc_async_read_barrier(noc);
        xfers<kZ, n_zc, false>(out_addr, stage, noc);
        if constexpr (n_copies > 0) {
            invalidate_l1_cache();
            const uint32_t last = copies<kElem, kC, n_copies>(stage, image);
            (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last & ~3u));
            xfers<kW, n_imgw, false>(out_addr, image, noc);
        }
        noc_async_write_barrier(noc);
    }
}
