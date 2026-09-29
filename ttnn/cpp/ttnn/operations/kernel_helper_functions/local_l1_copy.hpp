// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar-only local L1->L1 copy helper.
//
// On Quasar, relocating data already resident in L1 (e.g. splitting a contiguously-read row into the
// tile's face-tiled layout) cannot use the historical WH/BH idiom of a NoC "local loopback" read
// (async_read with the source pointed at the core's own coordinates): on the Quasar emulator a
// self-loopback (source coords == destination coords) spins on can_post or silently drops, leaving
// the destination unwritten. Because the producing read has already been barriered the bytes are
// present, so a plain scalar (RISC) copy is both correct and cheaper than a NoC round-trip. The DFB
// getters hand out the uncached L1 alias on Quasar DM; RISC access through it is coherent, so the
// addresses (as returned by e.g. DataflowBuffer::get_write_ptr()) are used as-is.
//
// Compiled only for Quasar; WH/BH keep their NoC-loopback path, so callers gate the call in
// `#ifdef ARCH_QUASAR`.

#ifdef ARCH_QUASAR

#include <cstdint>

#include "internal/risc_attribs.h"

/**
 * @brief Synchronously copy @p n_bytes already resident in this core's L1 from @p src_l1_addr to
 *        @p dst_l1_addr with a scalar RISC copy (no NoC).
 *
 * On return the copied bytes are globally visible. The trailing `fence` is required because these
 * stores go through the uncached L1 alias and a DFB `push_back()` only posts the credit — without it
 * a consumer could observe the credit before the relocated bytes have landed. `volatile` prevents
 * compiler elision but does not provide that hardware ordering.
 *
 * @param dst_l1_addr Destination L1 address.
 * @param src_l1_addr Source L1 address; the read that produced it must already be barriered.
 * @param n_bytes     Number of bytes to copy.
 */
inline void local_l1_copy(uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t n_bytes) {
    auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(src_l1_addr));
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(dst_l1_addr));
    const uint32_t words = n_bytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < words; ++i) {
        dst[i] = src[i];
    }
    // Byte tail for sizes that are not a multiple of 4.
    const uint32_t tail = words * sizeof(uint32_t);
    if (tail < n_bytes) {
        auto* src_b = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(static_cast<uintptr_t>(src_l1_addr));
        auto* dst_b = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(static_cast<uintptr_t>(dst_l1_addr));
        for (uint32_t i = tail; i < n_bytes; ++i) {
            dst_b[i] = src_b[i];
        }
    }
    // Order the uncached-alias stores above ahead of whatever posts the DFB credit next.
    __asm__ __volatile__("fence" ::: "memory");
}

#endif  // ARCH_QUASAR
