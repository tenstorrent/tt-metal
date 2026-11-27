// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace g6_concat {

// Tile storage: four 16x16 faces, face-major, row-major inside a face. Row y of a tile is two
// face_row_bytes pieces at face_offset(y, 0) and face_offset(y, 1).
constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t FACE_COLS_PER_TILE = 2;

template <uint32_t face_row_bytes>
FORCE_INLINE uint32_t face_offset(uint32_t row, uint32_t face_col) {
    constexpr uint32_t face_bytes = FACE_ROWS * face_row_bytes;
    return (FACE_COLS_PER_TILE * (row / FACE_ROWS) + face_col) * face_bytes + (row % FACE_ROWS) * face_row_bytes;
}

// Calls f(offset_bytes, size_bytes) for the contiguous pieces holding tile rows [0, valid_rows).
// A face half that is full is one piece (its two faces are adjacent); a partial half is one piece
// per face column.
template <uint32_t face_row_bytes, typename F>
FORCE_INLINE void for_each_valid_row_piece(uint32_t valid_rows, F&& f) {
    constexpr uint32_t face_bytes = FACE_ROWS * face_row_bytes;
    for (uint32_t half = 0; half < TILE_ROWS / FACE_ROWS; ++half) {
        const uint32_t half_first_row = half * FACE_ROWS;
        if (valid_rows <= half_first_row) {
            return;
        }
        const uint32_t rows = (valid_rows - half_first_row < FACE_ROWS) ? valid_rows - half_first_row : FACE_ROWS;
        const uint32_t half_offset = half * FACE_COLS_PER_TILE * face_bytes;
        if (rows == FACE_ROWS) {
            f(half_offset, FACE_COLS_PER_TILE * face_bytes);
        } else {
            for (uint32_t face_col = 0; face_col < FACE_COLS_PER_TILE; ++face_col) {
                f(half_offset + face_col * face_bytes, rows * face_row_bytes);
            }
        }
    }
}

// Walks output pages in row-major order and reports v(p), the valid rows of the current page.
// Only the last tile row of each image can be partial.
struct OutputRowCursor {
    uint32_t col;
    uint32_t tile_row;
    uint32_t out_wt;
    uint32_t out_ht;
    uint32_t last_row_valid_rows;

    FORCE_INLINE OutputRowCursor(uint32_t page, uint32_t out_wt_, uint32_t out_ht_, uint32_t last_row_valid_rows_) :
        col(page % out_wt_),
        tile_row((page / out_wt_) % out_ht_),
        out_wt(out_wt_),
        out_ht(out_ht_),
        last_row_valid_rows(last_row_valid_rows_) {}

    FORCE_INLINE uint32_t valid_rows() const { return tile_row + 1 == out_ht ? last_row_valid_rows : TILE_ROWS; }

    FORCE_INLINE void advance() {
        if (++col == out_wt) {
            col = 0;
            if (++tile_row == out_ht) {
                tile_row = 0;
            }
        }
    }
};

// trim_rows_max == 0 (bfloat8_b pages, ROW_MAJOR sticks) folds every page to a whole-page move at
// compile time, which also leaves the row cursor dead.
template <uint32_t trim_rows_max>
FORCE_INLINE bool is_trimmed(uint32_t valid_rows) {
    if constexpr (trim_rows_max == 0) {
        return false;
    } else {
        return valid_rows < TILE_ROWS && valid_rows <= trim_rows_max;
    }
}

}  // namespace g6_concat
