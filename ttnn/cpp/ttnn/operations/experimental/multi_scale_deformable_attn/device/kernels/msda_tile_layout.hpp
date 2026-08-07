// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace msda_tile_layout {

// Face holds FACE_ROWS rows, each row has D bf16 values.
// D is the per-query channel width (multiple of 16).
// For D <= 32: each face row holds one row of D values.
// For D > 32: each logical row spans multiple face-rows (bitsliced).
template <uint32_t D>
constexpr uint32_t FACE_ROWS = 16;
template <uint32_t D>
constexpr uint32_t FACE_NBYTES = FACE_ROWS<D> * D * sizeof(uint16_t);
template <uint32_t D>
constexpr uint32_t WITHIN_FACE_ROW_STRIDE = D * sizeof(uint16_t);

// Byte offsets (relative to the tile base in L1) for the low-half and
// high-half of tile row r.
//   lo: the cols-0..15 half (TL for r<16, BL for r>=16)
//   hi: the cols-16..31 half (TR for r<16, BR for r>=16)
template <uint32_t D>
struct RowOffsets {
    uint32_t lo;
    uint32_t hi;
};

template <uint32_t D>
inline RowOffsets<D> tile_row_offsets(uint32_t r) {
    if (r < 16) {
        return {r * WITHIN_FACE_ROW_STRIDE<D>, FACE_NBYTES<D> + r * WITHIN_FACE_ROW_STRIDE<D>};
    }
    const uint32_t rr = r - 16;
    return {2 * FACE_NBYTES<D> + rr * WITHIN_FACE_ROW_STRIDE<D>, 3 * FACE_NBYTES<D> + rr * WITHIN_FACE_ROW_STRIDE<D>};
}

// Byte offset for col 0 of tile row r. Only the low-half is needed for
// COL bcast scalar tiles (where only col 0 of each face is read).
template <uint32_t D>
inline uint32_t tile_col0_offset(uint32_t r) {
    const uint32_t face_base = (r < 16) ? 0u : (2u * FACE_NBYTES<D>);
    const uint32_t face_row = (r < 16) ? r : (r - 16);
    return face_base + face_row * WITHIN_FACE_ROW_STRIDE<D>;
}

}  // namespace msda_tile_layout
