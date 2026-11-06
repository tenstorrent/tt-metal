// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sparse 2D dispatch's shared kernel constants and span helpers. The mesh constants come in as defines:
//   SA_R, SA_C            mesh rows / columns
//   SA_S                  own rows a chip
//   SA_GMAX               (C - 1) + (R - 1) - 1: the last END generation
//   SA_RW                 route words per own row (1: mask | order << 16 | link << 17; 2: mask, order | link << 1)
//   SA_WRAP_X, SA_WRAP_Y  0 / 1: the axis is a ring
//   SA_XRING, SA_YRING    the ring span tables (ring_table(C, wrap_x), ring_table(R, wrap_y)) as comma lists
//
// Manifest entries are 16 bytes: word 0 = slot | hops left << 16 | generation << 20 | order << 24 (or END | g << 20),
// word 1 = the destination mask. A port core's semaphore words (`words`, 16 B apart): the rings' three entry counters,
// then the neighbour fence's ready word.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace sa {

constexpr uint32_t R = SA_R;
constexpr uint32_t C = SA_C;
constexpr uint32_t S = SA_S;
constexpr uint32_t GMAX = SA_GMAX;
constexpr bool WRAP_X = SA_WRAP_X != 0, WRAP_Y = SA_WRAP_Y != 0;
// ring spans by the destination mask rotated to the position (a ring looks the same from every chip: 2^n x 2 bytes, not
// n x 2^n x 2, which an 8-ring would put at 4 KB of the RISC's local memory); a line's span needs no table
constexpr uint8_t XRING[WRAP_X ? (1u << C) * 2 : 1] = {SA_XRING};
constexpr uint8_t YRING[WRAP_Y ? (1u << R) * 2 : 1] = {SA_YRING};
constexpr uint32_t END = 1u << 31;
constexpr uint32_t SKIP = 1u << 30;
constexpr uint32_t META_WORDS = 4;
constexpr uint32_t SEM_STRIDE = 16;  // bytes between the semaphore words
constexpr uint32_t READY_SEM = 3;    // the fence's ready word (after the three ring counters)

constexpr uint32_t RW = SA_RW;
FORCE_INLINE uint32_t route_mask(const uint32_t* w, uint32_t t) { return RW == 2 ? w[2 * t] : w[t] & 0xFFFF; }
FORCE_INLINE uint32_t route_flags(const uint32_t* w, uint32_t t) { return RW == 2 ? w[2 * t + 1] : w[t] >> 16; }
// ring k's first entry in the manifest (ring_offs: ring 1's | ring 2's << 16, in entries)
FORCE_INLINE uint32_t ring_off(uint32_t ring_offs, uint32_t k) {
    return k == 0 ? 0 : (ring_offs >> (16 * (k - 1))) & 0xFFFF;
}
FORCE_INLINE uint32_t tie_of(uint32_t t) { return ((t * 2654435761u) >> 30) & 1; }
// (a, b) packed a | b << 4: hops forward / backward from pos covering mask m's set positions (pos itself ignored)
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
        return (hi > pos ? hi - pos : 0) | ((lo < pos ? pos - lo : 0) << 4);
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

}  // namespace sa
