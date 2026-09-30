// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre — the single source of truth for the two in-L1 tile layouts the dataflow kernels touch.
//
// 1. Row-major (normal) 32x32 fp32 tile: element (row r, col c) lives in face
//    ((r >= 16) * 2 + (c >= 16)), at face-row (r & 15), face-col (c & 15).
// 2. Coefficient-major ("SoA") tile: the 32 SFPU vectors of a DEST tile. Slot k is the 32 datums
//    `sfpi::dst_reg[k]` / SFPLOAD address 2k touches; lane l is token row l of the token tile-row.
//    From the SFPLOAD lane layout (Row = (Addr & ~3) + Lane/8, Col = (Lane & 7)*2 + ((Addr & 2) ? 1 : 0),
//    face base address 16*f), with Addr = 2k:
//        face      = k >> 3
//        face_row  = 4 * ((k & 7) >> 1) + (l >> 3)
//        face_col  = 2 * (l & 7) + (k & 1)
//    Every per-token computation is then lane-wise: no cross-lane shuffles in the SFPU ops.

#pragma once

#include <stdint.h>

namespace mhc_layout {

constexpr uint32_t TILE_DATUMS = 1024;

// fp32-word index of element (row r, col c) in a row-major tile.
constexpr uint32_t rc_index(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

// fp32-word index of (slot k, lane l) in a coefficient-major tile.
constexpr uint32_t slot_index(uint32_t k, uint32_t l) {
    return (k >> 3) * 256u + (4u * ((k & 7u) >> 1) + (l >> 3)) * 16u + 2u * (l & 7u) + (k & 1u);
}

}  // namespace mhc_layout

namespace mhc_layout {

// Both maps are separable: rc_index(r, c) = rc_index(r, 0) + rc_index(0, c) and
// slot_index(k, l) = slot_index(k, 0) + slot_index(0, l). The movers below exploit that: the per-lane
// base is computed once per lane and the per-slot / per-column offsets are compile-time constants of
// a fully unrolled inner loop (one load + one store per datum, no index arithmetic).

// dst coefficient-major tile, slots [DST_SLOT0, DST_SLOT0 + COUNT) <- src row-major tile, cols [SRC_COL0, +COUNT)
template <uint32_t COUNT, uint32_t SRC_COL0, uint32_t DST_SLOT0>
inline __attribute__((always_inline)) void cols_to_slots(const float* __restrict src, float* __restrict dst) {
#pragma GCC unroll 1
    for (uint32_t l = 0; l < 32; ++l) {
        const float* s = src + rc_index(l, 0);
        float* d = dst + slot_index(0, l);
#pragma GCC unroll 32
        for (uint32_t k = 0; k < COUNT; ++k) {
            d[slot_index(DST_SLOT0 + k, 0)] = s[rc_index(0, SRC_COL0 + k)];
        }
    }
}

// dst row-major tile, cols [DST_COL0, DST_COL0 + COUNT) <- src coefficient-major tile, slots [SRC_SLOT0, +COUNT)
template <uint32_t COUNT, uint32_t SRC_SLOT0, uint32_t DST_COL0>
inline __attribute__((always_inline)) void slots_to_cols(const float* __restrict src, float* __restrict dst) {
#pragma GCC unroll 1
    for (uint32_t l = 0; l < 32; ++l) {
        const float* s = src + slot_index(0, l);
        float* d = dst + rc_index(l, 0);
#pragma GCC unroll 32
        for (uint32_t k = 0; k < COUNT; ++k) {
            d[rc_index(0, DST_COL0 + k)] = s[slot_index(SRC_SLOT0 + k, 0)];
        }
    }
}

}  // namespace mhc_layout
