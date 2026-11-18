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

// The first real step at or after j (k_cols: none).
template <uint32_t k_cols, uint32_t num_owning_cols>
inline uint32_t ring_next_real(uint32_t my_col, uint32_t j) {
    for (; j < k_cols; ++j) {
        if (nslice_owner<k_cols>(my_col, j) < num_owning_cols) {
            return j;
        }
    }
    return k_cols;
}

// Real steps j >= 1 of one M-block: each receives one partial from column my_col - 1.
template <uint32_t k_cols, uint32_t num_owning_cols>
inline uint32_t ring_incoming(uint32_t my_col) {
    uint32_t n = 0;
    for (uint32_t j = 1; j < k_cols; ++j) {
        n += nslice_owner<k_cols>(my_col, j) < num_owning_cols ? 1 : 0;
    }
    return n;
}

}  // namespace conv2d_3x3
