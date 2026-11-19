// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// N-slice-outer K split: the row's fp32 partials are reduce-scattered along a ring. Step j of the core in grid column
// my_col multiplies owner column my_col - 1 - j's N-slice (mod k_cols). The partial for owner o therefore starts on
// column o + 1 at step 0 and moves one column on per step, each core adding its own slice to it, until o adds its own
// slice last, at step k_cols - 1. Every hop is one column on the reader's NoC0 (+x), so no two partials of a row ever
// share a NoC link (a direct all-to-all puts up to k_cols / 2 sends on each link, and those waves were the walk's
// stall and the op's tail). Compute, writer (weights) and reader (forwarding) walk the same order.
// A step is real when its owner owns output columns (ring_is_owner). Every core of the row skips the same owners'
// whole chains, so a real step j >= 1 always receives the partial of step j - 1 from column my_col - 1.
// Owning columns: 0 .. num_owning_cols - 1, owner o summing N-slice o, unless the descriptor spreads them
// (RING_OWNER_MASK, the remote output store only): where C_out has fewer N-slices than the row has columns, the
// non-owning columns, together at the ring's end, made every chain pass all of them back to back before the column
// after them got its first partial; that column started one hop latency (~0.85 us: a 16 KB fp32 row group's NoC write
// plus its L1-acc pack) late per non-owner, and the lag held to its last step. Spread one per gap, each costs one hop.
// Owner o then sums N-slice ring_slice_of(o), its rank among the owners, and stores it through the remote output
// store. Measured (Blackhole p150b, LoFi, unzoned, median of 3): aspp.convs.3 23658 -> 22509 ns, aspp.convs.2 47044 ->
// 45959 ns. A moved owner starts its slice's fp32 sum at another column: the same partials added in another order,
// which matched the contiguous owners bit for bit on every case measured (bf8_b x bf8_b partial sums fit fp32's
// mantissa) but is not bit-identical by construction.

#pragma once

#include <cstdint>

namespace conv2d_3x3 {

template <uint32_t k_cols>
inline uint32_t nslice_owner(uint32_t my_col, uint32_t j) {
    return (my_col + 2 * k_cols - 1 - j) % k_cols;
}

// Column o ends a real chain (it sums and stores an N-slice).
template <uint32_t num_owning_cols>
inline bool ring_is_owner(uint32_t o) {
#ifdef RING_OWNER_MASK
    return ((RING_OWNER_MASK >> o) & 1u) != 0;
#else
    return o < num_owning_cols;
#endif
}

// The N-slice owner column o sums: its rank among the owning columns.
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
// RING_RANK_PARITY (nslice_step_shape): the short group first on odd owner ranks (ring_slice_of) instead of odd owner
// columns: with spread owners (RING_OWNER_MASK) a core's consecutive real steps alternate rank parity, not column
// parity (owners {0, 2, 4, 6} are all even).
inline uint32_t ring_parity_of(uint32_t owner) {
#ifdef RING_RANK_PARITY
    return ring_slice_of(owner) & 1;
#else
    return owner & 1;
#endif
}

template <uint32_t bm, uint32_t R>
inline uint32_t ring_group_row0(uint32_t owner, uint32_t g) {
    if constexpr (bm % R != 0) {
        if (ring_parity_of(owner)) {
            return g == 0 ? (bm / R) * R : (g - 1) * R;
        }
    }
    return g * R;
}

// ring_group_row0 over a slice of `rows` valid tile-rows known only at runtime (the work-balanced M split gives grid
// rows different tile-row counts under one CT block height): the same order, with the short group of `rows`.
template <uint32_t R>
inline uint32_t ring_group_row0_rt(uint32_t rows, uint32_t owner, uint32_t g) {
    if (rows % R != 0 && ring_parity_of(owner)) {
        return g == 0 ? (rows / R) * R : (g - 1) * R;
    }
    return g * R;
}

}  // namespace conv2d_3x3
