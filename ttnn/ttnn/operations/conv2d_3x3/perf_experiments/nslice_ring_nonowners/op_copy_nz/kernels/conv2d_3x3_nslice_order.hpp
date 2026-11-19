// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// N-slice-outer K split: the row's fp32 partials are reduce-scattered along a ring. Step j of the core in grid column
// my_col multiplies owner column my_col - 1 - j's N-slice (mod k_cols). The partial for owner o therefore starts on
// column o + 1 at step 0 and moves one column on per step, each core adding its own slice to it, until o adds its own
// slice last, at step k_cols - 1. Every hop is one column on the reader's NoC0 (+x), so no two partials of a row ever
// share a NoC link (a direct all-to-all puts up to k_cols / 2 sends on each link, and those waves were the walk's
// stall and the op's tail). Compute, writer (weights) and reader (forwarding) walk the same order.
// Owners past the last C_out tile are skipped: a step is real when its owner owns output columns. Every core of the
// row skips the same owners' whole chains, so a real step j >= 1 always receives the partial of step j - 1 from
// column my_col - 1.

#pragma once

#include <cstdint>

namespace conv2d_3x3 {

template <uint32_t k_cols>
inline uint32_t nslice_owner(uint32_t my_col, uint32_t j) {
    return (my_col + 2 * k_cols - 1 - j) % k_cols;
}

// NRN experiment (spread owners, not bit-identical): RING_OWNER_MASK names the columns whose chain is real (they end
// one, sum it and store one N-slice through the remote output store); column o carries N-slice ring_slice_of(o), its
// rank among the owners. Without it the owners are columns 0 .. num_owning_cols - 1 and column o carries N-slice o.
template <uint32_t num_owning_cols>
inline bool ring_is_owner(uint32_t o) {
#ifdef RING_OWNER_MASK
    return ((RING_OWNER_MASK >> o) & 1u) != 0;
#else
    return o < num_owning_cols;
#endif
}

inline uint32_t ring_slice_of(uint32_t o) {
#ifdef RING_OWNER_MASK
    return static_cast<uint32_t>(__builtin_popcount(static_cast<uint32_t>(RING_OWNER_MASK) & ((1u << o) - 1u)));
#else
    return o;
#endif
}

// The first real step at or after j (k_cols: none).
template <uint32_t k_cols, uint32_t num_owning_cols>
inline uint32_t ring_next_real(uint32_t my_col, uint32_t j) {
    for (; j < k_cols; ++j) {
        if (ring_is_owner<num_owning_cols>(nslice_owner<k_cols>(my_col, j))) {
            return j;
        }
    }
    return k_cols;
}

// Row groups of an owner slice [bm, out_n] (one ring hop and one token each): R rows, the last group shorter when R
// does not divide bm. Odd owners take the short group first: a core's consecutive steps alternate owner parity (except
// at the wrap when k_cols is odd), so each step starts at the matmul subblock height the previous one ended on. The
// order is a function of the owner only, so the two columns a partial moves between walk its groups alike.
template <uint32_t bm, uint32_t R>
inline uint32_t ring_group_row0(uint32_t owner, uint32_t g) {
    if constexpr (bm % R != 0) {
        if (owner & 1) {
            return g == 0 ? (bm / R) * R : (g - 1) * R;
        }
    }
    return g * R;
}

// ring_group_row0 over a slice of `rows` valid tile-rows known only at runtime (the work-balanced M split gives grid
// rows different tile-row counts under one CT block height): the same order, with the short group of `rows`.
template <uint32_t R>
inline uint32_t ring_group_row0_rt(uint32_t rows, uint32_t owner, uint32_t g) {
    if (rows % R != 0 && (owner & 1)) {
        return g == 0 ? (rows / R) * R : (g - 1) * R;
    }
    return g * R;
}

// Real steps j >= 1 of one M-block: each receives one partial from column my_col - 1.
template <uint32_t k_cols, uint32_t num_owning_cols>
inline uint32_t ring_incoming(uint32_t my_col) {
    uint32_t n = 0;
    for (uint32_t j = 1; j < k_cols; ++j) {
        n += ring_is_owner<num_owning_cols>(nslice_owner<k_cols>(my_col, j)) ? 1 : 0;
    }
    return n;
}

}  // namespace conv2d_3x3
