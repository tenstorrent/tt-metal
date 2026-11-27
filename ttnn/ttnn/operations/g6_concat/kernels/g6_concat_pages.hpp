// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Output-page geometry shared by the g6_concat reader and writer. Both kernels walk the
// same per-core page range with the same cursor and the same block-cutting rule, so they
// agree on every block boundary without exchanging anything but the CB handshake.

#pragma once

#include <cstdint>

namespace g6c {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

// Position of an output page in storage order: image, tile row inside the image, tile column.
// Advanced incrementally so the per-page walk needs no division.
struct OutPageCursor {
    uint32_t page;
    uint32_t image;
    uint32_t tile_row;
    uint32_t col;
    uint32_t tile_cols;
    uint32_t tile_rows_per_image;
    uint32_t rows_tail;

    OutPageCursor(uint32_t first_page, uint32_t tile_cols_, uint32_t tile_rows_per_image_, uint32_t rows_tail_) :
        page(first_page), tile_cols(tile_cols_), tile_rows_per_image(tile_rows_per_image_), rows_tail(rows_tail_) {
        const uint32_t global_row = first_page / tile_cols;
        col = first_page - global_row * tile_cols;
        image = global_row / tile_rows_per_image;
        tile_row = global_row - image * tile_rows_per_image;
    }

    // Rows of this output page that are data; the rest are tile padding and are never moved.
    uint32_t valid_rows() const {
        return (rows_tail != 0 && tile_row == tile_rows_per_image - 1) ? rows_tail : TILE_ROWS;
    }

    void advance() {
        ++page;
        if (++col == tile_cols) {
            col = 0;
            if (++tile_row == tile_rows_per_image) {
                tile_row = 0;
                ++image;
            }
        }
    }
};

// Pages in the next block: block_pages, the rest of the core's range, and (row_gather) the
// rest of the current output tile row, whichever is smallest.
template <uint32_t block_pages, bool cut_at_tile_rows>
inline uint32_t block_extent(uint32_t pages_left, const OutPageCursor& cursor) {
    uint32_t extent = pages_left < block_pages ? pages_left : block_pages;
    if constexpr (cut_at_tile_rows) {
        const uint32_t row_left = cursor.tile_cols - cursor.col;
        extent = extent < row_left ? extent : row_left;
    }
    return extent;
}

// Calls fn(byte_offset, bytes) for each contiguous piece of a tile page's first `rows` rows.
// Faces are 16x16 and stored 0,1 (rows 0-15) then 2,3 (rows 16-31).
template <uint32_t page_bytes, uint32_t face_row_bytes, typename Fn>
inline void for_each_valid_row_piece(uint32_t rows, Fn&& fn) {
    constexpr uint32_t face_bytes = FACE_ROWS * face_row_bytes;
    static_assert(page_bytes == 4 * face_bytes, "valid-row trimming assumes a 32x32 tile of four 16x16 faces");
    if (rows == TILE_ROWS) {
        fn(0, page_bytes);
    } else if (rows >= FACE_ROWS) {
        fn(0, 2 * face_bytes);
        const uint32_t lower_bytes = (rows - FACE_ROWS) * face_row_bytes;
        if (lower_bytes != 0) {
            fn(2 * face_bytes, lower_bytes);
            fn(3 * face_bytes, lower_bytes);
        }
    } else {
        const uint32_t upper_bytes = rows * face_row_bytes;
        fn(0, upper_bytes);
        fn(face_bytes, upper_bytes);
    }
}

}  // namespace g6c
