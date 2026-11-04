// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The dispatch prep's route words, parametrized by the RW_* defines.
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
// (dispatch prep's route words, one token at a time)
namespace rw_ {
constexpr uint32_t R = RW_R, C = RW_C, ME = RW_ME, K = RW_K, RW = RW_RW;
constexpr bool WRAP_X = RW_WRAPX, WRAP_Y = RW_WRAPY;
constexpr uint8_t XRING[WRAP_X ? (1u << C) * 2 : 1] = {RW_XRING_INIT};
constexpr uint8_t YRING[WRAP_Y ? (1u << R) * 2 : 1] = {RW_YRING_INIT};
constexpr uint8_t CHIP_OF[RW_E] = {RW_CHIPS_INIT};
inline uint32_t tie_of(uint32_t t) { return ((t * 2654435761u) >> 30) & 1; }
inline uint32_t hi_bit(uint32_t m) {
    uint32_t b = 0;
    while (m >>= 1) {
        ++b;
    }
    return b;
}
inline uint32_t lo_bit(uint32_t m) {
    uint32_t b = 0;
    while (!(m & 1)) {
        m >>= 1, ++b;
    }
    return b;
}
template <uint32_t n, bool wrap>
inline uint32_t span_of(const uint8_t* ring, uint32_t pos, uint32_t m, uint32_t t) {
    m &= ~(1u << pos) & ((1u << n) - 1);
    if (m == 0) {
        return 0;
    }
    if constexpr (wrap) {
        return ring[(((m >> pos) | (m << (n - pos))) & ((1u << n) - 1)) * 2 + tie_of(t)];
    } else {
        const uint32_t hi = hi_bit(m), lo = lo_bit(m);
        return (hi > pos ? hi - pos : 0) | ((lo < pos ? pos - lo : 0) << 4);
    }
}
inline uint32_t xspan(uint32_t pos, uint32_t cm, uint32_t t) { return span_of<C, WRAP_X>(XRING, pos, cm, t); }
inline uint32_t yspan(uint32_t pos, uint32_t rm, uint32_t t) { return span_of<R, WRAP_Y>(YRING, pos, rm, t); }
inline uint32_t links(uint32_t s) { return (s & 15) + (s >> 4); }
inline uint32_t cols_of(uint32_t m) {
    uint32_t cm = 0;
    for (uint32_t r = 0; r < R; ++r) {
        cm |= (m >> (r * C)) & ((1u << C) - 1);
    }
    return cm;
}
inline uint32_t rows_of(uint32_t m) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) {
        rm |= (((m >> (r * C)) & ((1u << C) - 1)) != 0) << r;
    }
    return rm;
}
inline uint32_t rows_in_col(uint32_t m, uint32_t c) {
    uint32_t rm = 0;
    for (uint32_t r = 0; r < R; ++r) {
        rm |= ((m >> (r * C + c)) & 1) << r;
    }
    return rm;
}
inline uint32_t cols_in_row(uint32_t m, uint32_t r) { return (m >> (r * C)) & ((1u << C) - 1); }
// token t's route word(s) (RW 1: mask | order << 16 | link << 17; RW 2: mask, order | link << 1) from its K expert
// ids, at o[RW t]
inline void put(volatile uint32_t* o, uint32_t t, uint32_t m, uint32_t flags) {
    if constexpr (RW == 2) {
        o[2 * t] = m;
        o[2 * t + 1] = flags;
    } else {
        o[t] = m | (flags << 16);
    }
}
inline void word(volatile uint32_t* o, uint32_t i, volatile const uint16_t* ids, uint32_t t) {
    const uint32_t r0 = ME / C, c0 = ME % C;
    uint32_t m = 0;
    for (uint32_t k = 0; k < K; ++k) {
        m |= 1u << CHIP_OF[ids[k]];
    }
    m &= ~(1u << ME);
    if (m == 0) {
        return put(o, i, 0, 0);
    }
    const uint32_t cm = cols_of(m), rm = rows_of(m);
    uint32_t hx = links(xspan(c0, cm & ~(1u << c0), t)), hy = links(yspan(r0, rm & ~(1u << r0), t));
    for (uint32_t c = 0; c < C; ++c) {
        if (((cm | (1u << c0)) >> c) & 1) {
            hx += links(yspan(r0, rows_in_col(m, c) & ~(1u << r0), t));
        }
    }
    for (uint32_t r = 0; r < R; ++r) {
        if (((rm | (1u << r0)) >> r) & 1) {
            hy += links(xspan(c0, cols_in_row(m, r) & ~(1u << c0), t));
        }
    }
    const uint32_t hsh = t * 2654435761u;
    const uint32_t order = hx < hy ? 0 : (hy < hx ? 1 : (hsh >> 31) & 1);
    put(o, i, m, order | (((hsh >> 29) & 1) << 1));
}
}  // namespace rw_
