// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Row / tile-index mapping shared by the Quasar layer_norm reader, compute and writer kernels, so the three
// sides of the STRIDED rings agree by construction (see the compute kernel header for the pass structure).
namespace qsr_layernorm {

// Row of NEO t in round k on a node that owns tile-rows [row_start, row_start + NCHt): its own row when that
// is a real row, else NEO 0's row of the same round (recomputed and re-stored as a byte-identical duplicate
// so every NEO, reader hart and writer hart runs the same number of rounds).
inline uint32_t row_of(uint32_t row_start, uint32_t NCHt, uint32_t C, uint32_t t, uint32_t k) {
    const uint32_t local = k * C + t;
    return row_start + (local < NCHt ? local : k * C);
}

// Source tile of pass-2 processing position q (0..Wt-1): the resident tail [Wt - tail, Wt) is processed
// first, then the reloaded head, i.e. src(q) = (Wt - tail + q) mod Wt.
template <uint32_t Wt, uint32_t tail>
constexpr uint32_t src_of(uint32_t q) {
    return q < tail ? (Wt - tail) + q : q - tail;
}

}  // namespace qsr_layernorm
