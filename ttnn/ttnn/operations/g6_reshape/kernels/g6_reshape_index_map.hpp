// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace g6 {

constexpr uint32_t TILE = 32;
constexpr uint32_t FACE = 16;

// Tile geometry of one TILE tensor. Leading dims fold into images of H rows; every divisor is a
// compile-time constant so the index math compiles to multiplies.
template <uint32_t W_, uint32_t H_, uint32_t Wt_, uint32_t Ht_>
struct Geometry {
    static constexpr uint32_t W = W_;
    static constexpr uint32_t H = H_;
    static constexpr uint32_t Wt = Wt_;
    static constexpr uint32_t Ht = Ht_;

    static constexpr uint32_t valid_rows(uint32_t tr) { return (H - TILE * tr) < TILE ? (H - TILE * tr) : TILE; }
    static constexpr uint32_t valid_cols(uint32_t tc) { return (W - TILE * tc) < TILE ? (W - TILE * tc) : TILE; }
};

// Byte offset of element (r, c) inside a 32x32 tile: four 16x16 faces, face-major, row-major in a face.
template <uint32_t esize>
FORCE_INLINE constexpr uint32_t face_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * (FACE * FACE * esize) + (r & (FACE - 1)) * (FACE * esize) +
           (c & (FACE - 1)) * esize;
}

// Tile id and intra-tile byte offset of the element at global row g (images folded), column c.
template <typename Geo, uint32_t esize>
FORCE_INLINE void locate(uint32_t g, uint32_t c, uint32_t& tile, uint32_t& off) {
    const uint32_t img = g / Geo::H;
    const uint32_t h = g - img * Geo::H;
    tile = (img * Geo::Ht + h / TILE) * Geo::Wt + c / TILE;
    off = face_off<esize>(h % TILE, c % TILE);
}

// Calls fn(off, bytes) once per merged byte run that holds rows [r_lo, r_hi) of one tile. Faces are
// visited in byte order; faces 1 and 3 only when the tile has more than 16 valid columns. Byte-adjacent
// face ranges merge, so 16 or 32 full-width rows move as a single run.
template <uint32_t esize, typename Fn>
FORCE_INLINE void for_each_tile_run(uint32_t r_lo, uint32_t r_hi, bool two_halves, Fn&& fn) {
    constexpr uint32_t face_row_bytes = FACE * esize;
    constexpr uint32_t face_bytes = FACE * face_row_bytes;
    uint32_t run_off = 0;
    uint32_t run_bytes = 0;
    for (uint32_t f = 0; f < 4; ++f) {
        if ((f & 1) && !two_halves) {
            continue;
        }
        const uint32_t f_r0 = (f >> 1) * FACE;
        const uint32_t lo = r_lo > f_r0 ? r_lo : f_r0;
        const uint32_t hi = r_hi < f_r0 + FACE ? r_hi : f_r0 + FACE;
        if (lo >= hi) {
            continue;
        }
        const uint32_t off = f * face_bytes + (lo - f_r0) * face_row_bytes;
        const uint32_t bytes = (hi - lo) * face_row_bytes;
        if (run_bytes != 0 && run_off + run_bytes == off) {
            run_bytes += bytes;
        } else {
            if (run_bytes != 0) {
                fn(run_off, run_bytes);
            }
            run_off = off;
            run_bytes = bytes;
        }
    }
    if (run_bytes != 0) {
        fn(run_off, run_bytes);
    }
}

// Position of a lane in the tile-major strip order of the tile-side tensor.
struct StripCursor {
    uint32_t tile;
    uint32_t img;
    uint32_t tr;
    uint32_t tc;
    uint32_t p;
};

template <typename Geo>
FORCE_INLINE StripCursor cursor_at(uint32_t tile, uint32_t p) {
    const uint32_t tc = tile % Geo::Wt;
    const uint32_t rest = tile / Geo::Wt;
    return {tile, rest / Geo::Ht, rest % Geo::Ht, tc, p};
}

// One tile of a block as the lane sees it: its slot index inside the block and the lane's rows.
struct BlockTile {
    uint32_t k;
    uint32_t tile;
    uint32_t img;
    uint32_t tr;
    uint32_t tc;
    uint32_t r_lo;
    uint32_t r_hi;
};

// Walks the tiles of one block (at most block_tiles consecutive tile-side tiles, fewer when the
// lane's strips run out) and advances the cursor and the remaining strip count past it.
template <typename Geo, uint32_t strip_rows, uint32_t block_tiles, typename Fn>
FORCE_INLINE void walk_block(StripCursor& cur, uint32_t& remaining, Fn&& fn) {
    for (uint32_t k = 0; k < block_tiles && remaining != 0; ++k) {
        const uint32_t v = Geo::valid_rows(cur.tr);
        const uint32_t strips = (v + strip_rows - 1) / strip_rows;
        const uint32_t avail = strips - cur.p;
        const uint32_t take = remaining < avail ? remaining : avail;
        const uint32_t r_lo = cur.p * strip_rows;
        const uint32_t r_end = (cur.p + take) * strip_rows;
        fn(BlockTile{k, cur.tile, cur.img, cur.tr, cur.tc, r_lo, r_end < v ? r_end : v});
        remaining -= take;
        if (take == avail) {
            cur.p = 0;
            ++cur.tile;
            if (++cur.tc == Geo::Wt) {
                cur.tc = 0;
                if (++cur.tr == Geo::Ht) {
                    cur.tr = 0;
                    ++cur.img;
                }
            }
        } else {
            cur.p += take;
        }
    }
}

}  // namespace g6
