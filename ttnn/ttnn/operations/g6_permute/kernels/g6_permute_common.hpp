// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Regime rows_tile: out[n, b, a, w] = x[n, a, b, w] on bfloat16 tiles of four 16x16 faces.
// Work unit = one output tile (n, b, ta, tw), ordered b-fastest:
//     u = ((n * tensor_a_tiles + ta) * tensor_w_tiles + tw) * B + b
// The block rule below is mirrored by _block_extent in g6_permute_program_descriptor.py.
namespace g6_permute {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

// Byte offsets inside one tile, derived from the 32-B (bfloat16) face-row segment.
template <uint32_t face_row_bytes>
struct TileGeometry {
    static constexpr uint32_t face_bytes = FACE_ROWS * face_row_bytes;
    static constexpr uint32_t band_bytes = 2 * face_bytes;  // 16 rows, both column halves
    static constexpr uint32_t tile_bytes = 2 * band_bytes;

    // Offset of tile row r (0..31), column half h (0, 1).
    static constexpr uint32_t off(uint32_t r, uint32_t h) {
        return (r / FACE_ROWS) * band_bytes + h * face_bytes + (r % FACE_ROWS) * face_row_bytes;
    }
};

struct ByteRange {
    uint32_t offset;
    uint32_t len;
};

// Contiguous byte ranges covering tile rows [r0, r1): two face ranges per touched band, merged
// into one band range when the band is fully covered, and one tile range for the whole tile.
// Returns the range count (<= 4).
template <uint32_t face_row_bytes>
inline uint32_t row_span_ranges(uint32_t r0, uint32_t r1, ByteRange ranges[4]) {
    using G = TileGeometry<face_row_bytes>;
    if (r0 == 0 && r1 == TILE_ROWS) {
        ranges[0] = {0, G::tile_bytes};
        return 1;
    }
    uint32_t count = 0;
    for (uint32_t band = 0; band < 2; ++band) {
        const uint32_t band_r0 = band * FACE_ROWS;
        const uint32_t lo = r0 > band_r0 ? r0 : band_r0;
        const uint32_t hi = r1 < band_r0 + FACE_ROWS ? r1 : band_r0 + FACE_ROWS;
        if (lo >= hi) {
            continue;
        }
        if (lo == band_r0 && hi == band_r0 + FACE_ROWS) {
            ranges[count++] = {band * G::band_bytes, G::band_bytes};
        } else {
            const uint32_t len = (hi - lo) * face_row_bytes;
            ranges[count++] = {G::off(lo, 0), len};
            ranges[count++] = {G::off(lo, 1), len};
        }
    }
    return count;
}

// Walks one core's unit range block by block.
template <uint32_t B, uint32_t tensor_a_tiles, uint32_t tensor_w_tiles, uint32_t block_b_rows>
struct BlockCursor {
    static_assert(block_b_rows >= 1 && block_b_rows <= TILE_ROWS, "a block stays inside one b-tile");

    uint32_t n, ta, tw, b;  // first unit of the current block
    uint32_t remaining;     // units left in this core's range

    BlockCursor(uint32_t start_unit, uint32_t num_units) : remaining(num_units) {
        b = start_unit % B;
        uint32_t rest = start_unit / B;
        tw = rest % tensor_w_tiles;
        rest /= tensor_w_tiles;
        ta = rest % tensor_a_tiles;
        n = rest / tensor_a_tiles;
    }

    bool done() const { return remaining == 0; }

    // Block rule: stay in the core's range, keep (n, ta, tw), keep b div 32, at most block_b_rows.
    uint32_t extent() const {
        uint32_t run = block_b_rows;
        run = remaining < run ? remaining : run;
        run = (B - b) < run ? (B - b) : run;
        const uint32_t to_tile_end = TILE_ROWS - b % TILE_ROWS;
        return to_tile_end < run ? to_tile_end : run;
    }

    // Live rows of output tile row ta (= live input tiles of the block).
    uint32_t a_live(uint32_t A) const {
        const uint32_t left = A - ta * TILE_ROWS;
        return left < TILE_ROWS ? left : TILE_ROWS;
    }

    void advance(uint32_t run) {
        remaining -= run;
        b += run;
        if (b == B) {
            b = 0;
            if (++tw == tensor_w_tiles) {
                tw = 0;
                if (++ta == tensor_a_tiles) {
                    ta = 0;
                    ++n;
                }
            }
        }
    }
};

// Per-step profiler zones are kept for the first blocks of a core only: zone executions are
// capped at ~125 per RISC and a zone past the cap vanishes silently.
constexpr uint32_t ZONED_BLOCKS = 16;

}  // namespace g6_permute

#define G6_ZONED_STEP(zoned, name, ...) \
    if constexpr (zoned) {              \
        DeviceZoneScopedN(name);        \
        __VA_ARGS__;                    \
    } else {                            \
        __VA_ARGS__;                    \
    }
