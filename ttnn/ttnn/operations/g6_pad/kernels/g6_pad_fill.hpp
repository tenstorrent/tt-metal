// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Constant fills of L1 byte ranges for g6_pad's fill pages and masks.
//
// Helper bypass: dataflow_kernel_lib::fill_l1_range (kernel_lib/l1_helpers.hpp) is not used for
// the stores. Its bulk loop issues one volatile word store per iteration, measured at 1.19 us per
// 2 KB bf16 tile on BRISC against 0.51 us for the 8-wide loop below, and these fills sit on the
// critical path of every boundary and fill-only core. Unrolling the helper's bulk loop would
// close the gap.

#pragma once

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

namespace g6_pad {

// Stores the element pattern over [addr, addr + n_bytes). addr and n_bytes are multiples of
// elem_bytes; a 2-byte element range may start or end on a half word.
template <uint32_t elem_bytes>
FORCE_INLINE void store_constant(uint32_t addr, uint32_t n_bytes, uint32_t fill_bits) {
    static_assert(elem_bytes == 2 || elem_bytes == 4, "g6_pad fills 2- or 4-byte elements");
    uint32_t end = addr + n_bytes;
    uint32_t word = fill_bits;
    if constexpr (elem_bytes == 2) {
        const uint32_t half = fill_bits & 0xFFFF;
        word = half | (half << 16);
        if ((addr & 2) != 0 && addr < end) {
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr) = half;
            addr += 2;
        }
        if ((end & 2) != 0 && end > addr) {
            end -= 2;
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(end) = half;
        }
    }
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t n_words = (end - addr) >> 2;
    uint32_t i = 0;
    for (; i + 8 <= n_words; i += 8) {
        p[i] = word;
        p[i + 1] = word;
        p[i + 2] = word;
        p[i + 3] = word;
        p[i + 4] = word;
        p[i + 5] = word;
        p[i + 6] = word;
        p[i + 7] = word;
    }
    for (; i < n_words; ++i) {
        p[i] = word;
    }
}

// store_constant with the length and the start's alignment known at compile time: no runtime
// head/tail tests, fully unrolled. pattern_word is the element pattern replicated to 32 bits.
template <uint32_t elem_bytes, uint32_t n_bytes, uint32_t start_mod_4>
FORCE_INLINE void store_constant_fixed(uint32_t addr, uint32_t pattern_word) {
    static_assert(elem_bytes == 2 || elem_bytes == 4, "g6_pad fills 2- or 4-byte elements");
    static_assert(n_bytes % elem_bytes == 0 && start_mod_4 % elem_bytes == 0, "whole elements only");
    constexpr bool head = (start_mod_4 & 2) != 0 && n_bytes >= 2;
    constexpr uint32_t body_bytes = head ? n_bytes - 2 : n_bytes;
    constexpr uint32_t n_words = body_bytes / 4;
    constexpr bool tail = (body_bytes % 4) != 0;
    if constexpr (head) {
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr) = static_cast<uint16_t>(pattern_word);
        addr += 2;
    }
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
#pragma GCC unroll 16
    for (uint32_t i = 0; i < n_words; ++i) {
        p[i] = pattern_word;
    }
    if constexpr (tail) {
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr + n_words * 4) = static_cast<uint16_t>(pattern_word);
    }
}

template <uint32_t elem_bytes>
FORCE_INLINE uint32_t pattern_word(uint32_t fill_bits) {
    if constexpr (elem_bytes == 2) {
        const uint32_t half = fill_bits & 0xFFFF;
        return half | (half << 16);
    }
    return fill_bits;
}

// Fills bytes [offset, offset + n_bytes) past cb_id's write pointer with the element pattern c.
// c = 0 (every production extension call, most new-storage calls) goes through the NoC zero path,
// a loopback read of the firmware zero region, so the RISC-V issues one request instead of the
// stores. offset must be 16-B aligned (NoC L1 read alignment).
template <uint32_t elem_bytes>
FORCE_INLINE void fill_constant(uint32_t cb_id, uint32_t offset, uint32_t n_bytes, uint32_t fill_bits) {
    if (n_bytes == 0) {
        return;
    }
    if (fill_bits == 0) {
        Noc noc;
        CircularBuffer cb(cb_id);
        noc.async_write_zeros(cb, n_bytes, {.offset_bytes = offset});
        noc.write_zeros_l1_barrier();
    } else {
        store_constant<elem_bytes>(get_write_ptr(cb_id) + offset, n_bytes, fill_bits);
    }
}

}  // namespace g6_pad
