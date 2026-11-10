// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sparse 2D dispatch's shared kernel constants and span helpers. The mesh constants come in as defines:
//   SPARSE_DISPATCH_R, SPARSE_DISPATCH_C    mesh rows / columns
//   SPARSE_DISPATCH_S                       own rows a chip
//   SPARSE_DISPATCH_GMAX                    (C - 1) + (R - 1) - 1: the last END generation
//   SPARSE_DISPATCH_RW                      route words per own row (1: mask | order << 16 | link << 17;
//                                           2: mask, order | link << 1)
//   SPARSE_DISPATCH_WRAP_X, _WRAP_Y         0 / 1: the axis is a ring
//   SPARSE_DISPATCH_XRING, _YRING           the ring span tables (ring_table(C, wrap_x), ring_table(R, wrap_y)) as
//                                           comma lists
//
// Manifest entries are 16 bytes: word 0 = slot, hops left, generation, order (or END and the generation), word 1 = the
// destination mask (sparse_dispatch_2d_layout.hpp has the fields). A port core's semaphore words (`words`, 16 B apart):
// the rings' three entry counters, then the neighbour fence's ready word.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/sparse_dispatch_2d/device/kernels/sparse_dispatch_2d_layout.hpp"

namespace sparse_dispatch {

namespace lay = sparse_dispatch_2d_layout;

// A port core of another chip only (one program for the mesh, common/chip_program.hpp): the reader's direction arg
// or the sender's ready x is this, and the kernel returns.
constexpr uint32_t IDLE_PORT = lay::kIdlePort;

constexpr uint32_t R = SPARSE_DISPATCH_R;
constexpr uint32_t C = SPARSE_DISPATCH_C;
constexpr uint32_t S = SPARSE_DISPATCH_S;
constexpr uint32_t GMAX = SPARSE_DISPATCH_GMAX;
constexpr bool WRAP_X = SPARSE_DISPATCH_WRAP_X != 0, WRAP_Y = SPARSE_DISPATCH_WRAP_Y != 0;
// ring spans by the destination mask rotated to the position (a ring looks the same from every chip: 2^n x 2 bytes, not
// n x 2^n x 2, which an 8-ring would put at 4 KB of the RISC's local memory); a line's span needs no table
constexpr uint8_t XRING[WRAP_X ? (1u << C) * 2 : 1] = {SPARSE_DISPATCH_XRING};
constexpr uint8_t YRING[WRAP_Y ? (1u << R) * 2 : 1] = {SPARSE_DISPATCH_YRING};
constexpr uint32_t END = lay::kEndBit;
constexpr uint32_t SKIP = lay::kSkipBit;
constexpr uint32_t META_WORDS = lay::kEntryWords;
constexpr uint32_t SEM_STRIDE = 16;  // bytes between the semaphore words
constexpr uint32_t READY_SEM = 3;    // the fence's ready word (after the three ring counters)

constexpr uint32_t RW = SPARSE_DISPATCH_RW;
FORCE_INLINE uint32_t route_mask(const uint32_t* w, uint32_t t) {
    return RW == 2 ? w[2 * t] : lay::kMaskInWord.get(w[t]);
}
FORCE_INLINE uint32_t route_flags(const uint32_t* w, uint32_t t) {
    return RW == 2 ? w[2 * t + 1] : lay::kFlagsInWord.get(w[t]);
}
FORCE_INLINE uint32_t tie_of(uint32_t t) { return ((t * lay::kHashMultiplier) >> lay::kRingTieBit) & 1; }
// hops forward / backward (lay::span_byte) from pos covering mask m's set positions (pos itself ignored)
template <uint32_t n, bool wrap>
FORCE_INLINE uint32_t span_of(const uint8_t* ring, uint32_t pos, uint32_t m, uint32_t t) {
    m &= ~(1u << pos) & ((1u << n) - 1);
    if (m == 0) {
        return 0;
    }
    if constexpr (wrap) {
        return ring[(((m >> pos) | (m << (n - pos))) & ((1u << n) - 1)) * 2 + tie_of(t)];
    } else {
        const uint32_t hi = 31 - __builtin_clz(m), lo = __builtin_ctz(m);
        return lay::span_byte(hi > pos ? hi - pos : 0, lo < pos ? pos - lo : 0);
    }
}
FORCE_INLINE uint32_t xspan(uint32_t pos, uint32_t cm, uint32_t t) { return span_of<C, WRAP_X>(XRING, pos, cm, t); }
FORCE_INLINE uint32_t yspan(uint32_t pos, uint32_t rm, uint32_t t) { return span_of<R, WRAP_Y>(YRING, pos, rm, t); }
// columns holding a destination (any row)
FORCE_INLINE uint32_t cols_of(uint32_t m) {
    uint32_t cm = 0;
    for (uint32_t r = 0; r < R; ++r) {
        cm |= (m >> (r * C)) & ((1u << C) - 1);
    }
    return cm;
}
// rows holding a destination (any column)
FORCE_INLINE uint32_t rows_of(uint32_t m) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) {
        rm |= (((m >> (r * C)) & ((1u << C) - 1)) != 0) << r;
    }
    return rm;
}
// destinations in column c, as a row mask
FORCE_INLINE uint32_t rows_in_col(uint32_t m, uint32_t c) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) {
        rm |= ((m >> (r * C + c)) & 1) << r;
    }
    return rm;
}
// destinations in row r, as a column mask
FORCE_INLINE uint32_t cols_in_row(uint32_t m, uint32_t r) { return (m >> (r * C)) & ((1u << C) - 1); }

// The reader (own rows) and the sender (what the next chip does with a row) must compute the same spans, or the tree
// breaks: both go through these two.
// the span of a row's first leg along an axis from chip (r, c): over every column (or row) holding a destination
FORCE_INLINE uint32_t first_leg_span(bool axis_x, uint32_t r, uint32_t c, uint32_t m, uint32_t t) {
    return axis_x ? xspan(c, cols_of(m), t) : yspan(r, rows_of(m), t);
}
// the span of a row's turn at chip (r, c) into an axis: over the destinations in the chip's own row (or column)
FORCE_INLINE uint32_t turn_span(bool into_x, uint32_t r, uint32_t c, uint32_t m, uint32_t t) {
    return into_x ? xspan(c, cols_in_row(m, r), t) : yspan(r, rows_in_col(m, c), t);
}

}  // namespace sparse_dispatch
