// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Row geometry of a 32x32 tile: four 16x16 faces stored face-major, row-major inside a face
// (face 0 rows 0-15 cols 0-15 | face 1 rows 0-15 cols 16-31 | face 2 rows 16-31 cols 0-15 |
// face 3 rows 16-31 cols 16-31). Row y of a tile is two `row_bytes` pieces, one per face column.
namespace g6c {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

// Byte offset of tile row `row` in face column `face_col` (0: cols 0-15, 1: cols 16-31).
FORCE_INLINE uint32_t face_row_offset(uint32_t row, uint32_t face_col, uint32_t row_bytes) {
    return ((row / FACE_ROWS) * 2 + face_col) * FACE_ROWS * row_bytes + (row % FACE_ROWS) * row_bytes;
}

// Calls fn(offset, bytes) for each contiguous span holding rows [0, valid_rows) of a tile,
// 0 < valid_rows < 32: two spans in the top half, or the whole top half plus two in the bottom.
template <typename Fn>
FORCE_INLINE void for_each_valid_row_span(uint32_t valid_rows, uint32_t row_bytes, Fn&& fn) {
    const uint32_t face_bytes = FACE_ROWS * row_bytes;
    if (valid_rows < FACE_ROWS) {
        fn(0, valid_rows * row_bytes);
        fn(face_bytes, valid_rows * row_bytes);
        return;
    }
    const uint32_t bottom_rows = valid_rows - FACE_ROWS;
    if (bottom_rows == 0) {
        fn(0, 2 * face_bytes);
        return;
    }
    // Faces 0 and 1 whole, then face 2's first rows: one contiguous span.
    fn(0, 2 * face_bytes + bottom_rows * row_bytes);
    fn(3 * face_bytes, bottom_rows * row_bytes);
}

// Tile row and column of the current output page, advanced one page at a time in row-major
// page order; yields the page's valid rows without dividing.
struct OutputRowCursor {
    uint32_t tile_row;
    uint32_t col;
    uint32_t tiles_per_row;
    uint32_t tile_rows;
    uint32_t last_valid_rows;

    FORCE_INLINE uint32_t valid_rows() const { return tile_row == tile_rows - 1 ? last_valid_rows : TILE_ROWS; }

    FORCE_INLINE void advance() {
        if (++col == tiles_per_row) {
            col = 0;
            if (++tile_row == tile_rows) {
                tile_row = 0;
            }
        }
    }
};

// True when a page with `valid_rows` rows moves only those rows.
template <uint32_t trim_rows_max>
FORCE_INLINE bool trimmed(uint32_t valid_rows) {
    if constexpr (trim_rows_max == 0) {
        return false;
    } else {
        return valid_rows <= trim_rows_max && valid_rows < TILE_ROWS;
    }
}

}  // namespace g6c
