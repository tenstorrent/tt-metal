// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace msda_tile_layout {

// bf16 32x32 tile = 4 faces of 16x16 (each face = 512 B). One tile row
// spans two faces: the low half (cols 0..15) and the high half
// (cols 16..31). For row r ∈ [0, 16) those halves live in TL (offset 0)
// and TR (offset 512); for row r ∈ [16, 32) they live in BL (offset
// 1024) and BR (offset 1536). The within-face row stride is 32 B.
constexpr uint32_t FACE_NBYTES = 512;
constexpr uint32_t WITHIN_FACE_ROW_STRIDE = 32;

// Byte offsets (relative to the tile base in L1) for the low-half and
// high-half of tile row r.
//   lo: the cols-0..15 half (TL for r<16, BL for r>=16)
//   hi: the cols-16..31 half (TR for r<16, BR for r>=16)
struct RowOffsets {
    uint32_t lo;
    uint32_t hi;
};

inline RowOffsets tile_row_offsets(uint32_t r) {
    if (r < 16) {
        return {r * WITHIN_FACE_ROW_STRIDE, FACE_NBYTES + r * WITHIN_FACE_ROW_STRIDE};
    }
    const uint32_t rr = r - 16;
    return {2 * FACE_NBYTES + rr * WITHIN_FACE_ROW_STRIDE, 3 * FACE_NBYTES + rr * WITHIN_FACE_ROW_STRIDE};
}

// Byte offset for col 0 of tile row r. Only the low-half is needed for
// COL bcast scalar tiles (where only col 0 of each face is read).
inline uint32_t tile_col0_offset(uint32_t r) {
    const uint32_t face_base = (r < 16) ? 0u : (2u * FACE_NBYTES);
    const uint32_t face_row = (r < 16) ? r : (r - 16);
    return face_base + face_row * WITHIN_FACE_ROW_STRIDE;
}

}  // namespace msda_tile_layout
