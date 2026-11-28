// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tile_face_skip bench: which bytes of a 32x32 tile hold logical data.
//
// A tile is [hdr][face0 TL][face1 TR][face2 BL][face3 BR] (hdr = 0 for bf16/fp32/int formats, 64 B of shared
// exponents for bfloat8_b). When the logical W of the last tile column has W % 32 in 1..16, faces 1 and 3 of
// every last-column tile are pure padding; when H % 32 is in 1..16, faces 2 and 3 of every last-row tile are.
// Op contract (padding-in-ttnn §1): output padding may be left untouched and input padding must not reach live
// values -- typecast is elementwise, so an unread input face only feeds an unwritten output face.
//
// for_live_segments() calls fn(offset_bytes, bytes) for each maximal contiguous live run of the tile:
//   full tile                 -> [0, tile)
//   bottom faces dead         -> [0, hdr + 2 face)                (faces 0,1 are adjacent)
//   right faces dead          -> [0, hdr + face), [hdr + 2 face, hdr + 3 face)
//   both                      -> [0, hdr + face)
// Every offset is a multiple of 64 B for all formats used (face = 256/512/1024 B, hdr = 0/64 B), so DRAM
// (64 B) and L1 (16 B) NoC alignment hold for both ends.
#pragma once

#include <cstdint>

namespace tfs {

struct TilePos {
    uint32_t col;  // tile column inside the tile row, 0..Wt-1
    uint32_t row;  // tile row inside one batch slice, 0..Ht-1
};

template <uint32_t Wt, uint32_t Ht>
FORCE_INLINE TilePos tile_pos(uint32_t page) {  // once per core (one division)
    const uint32_t q = page / Wt;
    return {page - q * Wt, q % Ht};
}

template <uint32_t Wt, uint32_t Ht>
FORCE_INLINE void advance(TilePos& tp) {
    if (++tp.col == Wt) {
        tp.col = 0;
        if (++tp.row == Ht) {
            tp.row = 0;
        }
    }
}

template <
    uint32_t tile_bytes,
    uint32_t hdr,
    uint32_t face,
    bool skip_r_possible,
    bool skip_b_possible,
    uint32_t Wt,
    uint32_t Ht,
    typename F>
FORCE_INLINE void for_live_segments(const TilePos& tp, F&& fn) {
    const bool skip_r = skip_r_possible && (Wt == 1 || tp.col == Wt - 1);
    const bool skip_b = skip_b_possible && (Ht == 1 || tp.row == Ht - 1);
    if (!skip_r) {
        if (!skip_b) {
            fn(0u, tile_bytes);
        } else {
            fn(0u, hdr + 2 * face);
        }
    } else {
        fn(0u, hdr + face);
        if (!skip_b) {
            fn(hdr + 2 * face, face);
        }
    }
}

}  // namespace tfs
