// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Deliberately a separate copy of the equivalent helper in
// operations/experimental/multi_scale_deformable_attn/device/kernels/. That op
// is a fixed reference point for comparison and must not be modified, and a
// shared header would couple the two ops' kernel builds. The namespace differs
// so both can appear in one unity build.
// TODO(#45742): fold both into a shared kernel header once the per-op copies of
// the bf16 helpers are consolidated too.
namespace fused_msda_tile_layout {

// bf16 32x32 tile = 4 faces of 16x16 (each face = 512 B). One tile row spans
// two faces: the low half (cols 0..15) and the high half (cols 16..31). For row
// r in [0, 16) those halves live in TL (offset 0) and TR (offset 512); for row
// r in [16, 32) they live in BL (offset 1024) and BR (offset 1536). The
// within-face row stride is 32 B.
constexpr uint32_t FACE_NBYTES = 512;
constexpr uint32_t WITHIN_FACE_ROW_STRIDE = 32;
constexpr uint32_t TILE_NBYTES = 2048;
constexpr uint32_t TILE_MAX_ROWS = 32;

// Byte offsets (relative to the tile base in L1) for the low-half and high-half
// of tile row r.
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

// Byte offset for col 0 of tile row r. Only the low half is needed for COL
// bcast scalar tiles, where only col 0 of each face is read.
inline uint32_t tile_col0_offset(uint32_t r) {
    const uint32_t face_base = (r < 16) ? 0u : (2u * FACE_NBYTES);
    const uint32_t face_row = (r < 16) ? r : (r - 16);
    return face_base + face_row * WITHIN_FACE_ROW_STRIDE;
}

// Byte offset for col 16 of tile row r: col 0 of the face to the right, which
// the SFPU sees in the same lane as col 0.
inline uint32_t tile_col16_offset(uint32_t r) { return tile_col0_offset(r) + FACE_NBYTES; }

// float32 32x32 tile, same four-face order. Each element is 4 bytes, so a face
// is 1024 B and a face-row is 64 B. geom_x/geom_y use this when the reference
// point is float32: the SFPU multiplies that point, and a bf16 tile would have
// rounded it first.
constexpr uint32_t FP32_FACE_NBYTES = 1024;
constexpr uint32_t FP32_WITHIN_FACE_ROW_STRIDE = 64;
constexpr uint32_t FP32_TILE_NBYTES = 4096;

inline uint32_t tile_col0_offset_fp32(uint32_t r) {
    const uint32_t face_base = (r < 16) ? 0u : (2u * FP32_FACE_NBYTES);
    const uint32_t face_row = (r < 16) ? r : (r - 16);
    return face_base + face_row * FP32_WITHIN_FACE_ROW_STRIDE;
}

inline uint32_t tile_col16_offset_fp32(uint32_t r) { return tile_col0_offset_fp32(r) + FP32_FACE_NBYTES; }

// Byte offset of element (r, c) of a bf16 tile.
inline uint32_t tile_elem_offset(uint32_t r, uint32_t c) {
    const auto off = tile_row_offsets(r);
    return (c < 16 ? off.lo : off.hi) + (c % 16) * 2;
}

}  // namespace fused_msda_tile_layout
