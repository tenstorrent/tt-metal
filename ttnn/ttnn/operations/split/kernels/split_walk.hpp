// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The walk of split, shared by the reader and the writer so that both derive the identical sequence of units
// from the same runtime args (op_design.md, Block schedule). The host derives every geometry field once
// (split_program_artifacts.py, derive_geometry) and passes them as common runtime args.

#pragma once

#include <cstdint>

// The regime is the define SPLIT_REGIME, set by split_program_artifacts.py (REGIME_IDS): a preprocessor value and
// not a compile-time arg, because only the tile_row_shift reader binds the scratchpad window.
#define SPLIT_REGIME_PAGE_REMAP 0
#define SPLIT_REGIME_TILE_ROW_SHIFT 1

namespace split {


constexpr uint32_t TILE_ROWS = 32;

inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// page_remap: a unit is one input page u = (o * cut_pages + p) * inner_pages + i, which is exactly one page of
// chunk c = p / chunk_pages, at page-row p - c * chunk_pages of that chunk.
struct PageRemapWalk {
    uint32_t cut_pages, inner_pages, chunk_pages;
    uint32_t o = 0, p = 0, i = 0;  // input coordinates
    uint32_t c = 0, pc = 0;        // chunk, page-row inside the chunk

    PageRemapWalk(uint32_t cut, uint32_t inner, uint32_t chunk) : cut_pages(cut), inner_pages(inner), chunk_pages(chunk) {}

    void seek(uint32_t u) {
        i = u % inner_pages;
        const uint32_t row = u / inner_pages;
        p = row % cut_pages;
        o = row / cut_pages;
        c = p / chunk_pages;
        pc = p - c * chunk_pages;
    }
    uint32_t input_page() const { return (o * cut_pages + p) * inner_pages + i; }
    uint32_t chunk() const { return c; }
    uint32_t chunk_extent() const { return min_u32(chunk_pages, cut_pages - c * chunk_pages); }
    uint32_t output_page() const { return (o * chunk_extent() + pc) * inner_pages + i; }
    void next() {
        if (++i < inner_pages) {
            return;
        }
        i = 0;
        if (++p == cut_pages) {
            p = 0, pc = 0, c = 0, ++o;
        } else if (++pc == chunk_pages) {
            pc = 0, ++c;
        }
    }
    void advance(uint32_t n) {
        for (uint32_t k = 0; k < n; ++k) {
            next();
        }
    }
};

// tile_row_shift: a unit is one output tile, walked column-major (o, tw, c, j): for each input tile column
// (o, tw), the output tile-rows of chunk 0, then chunk 1, ... A column holds column_tiles output tiles.
// Output tile (o, tw, c, j) takes its valid rows from input rows [first_row(), first_row() + valid_rows()).
struct TileRowShiftWalk {
    uint32_t h_tiles, w_tiles, split_rows, cut_rows, chunk_tiles, column_tiles;
    uint32_t col = 0;  // o * w_tiles + tw
    uint32_t k = 0;    // position inside the column, = c * chunk_tiles + j
    uint32_t c = 0, j = 0;

    TileRowShiftWalk(uint32_t h, uint32_t w, uint32_t split, uint32_t cut, uint32_t ct, uint32_t col_t) :
        h_tiles(h), w_tiles(w), split_rows(split), cut_rows(cut), chunk_tiles(ct), column_tiles(col_t) {}

    void seek(uint32_t u) {
        col = u / column_tiles;
        k = u - col * column_tiles;
        c = k / chunk_tiles;
        j = k - c * chunk_tiles;
    }
    uint32_t chunk() const { return c; }
    uint32_t chunk_rows() const { return min_u32(split_rows, cut_rows - c * split_rows); }
    uint32_t chunk_tile_rows() const { return (chunk_rows() + TILE_ROWS - 1) / TILE_ROWS; }
    uint32_t first_row() const { return c * split_rows + TILE_ROWS * j; }
    uint32_t valid_rows() const { return min_u32(TILE_ROWS, chunk_rows() - TILE_ROWS * j); }
    uint32_t tiles_left_in_column() const { return column_tiles - k; }
    // Input tile at tile-row `ta` of this column.
    uint32_t input_page(uint32_t ta) const {
        const uint32_t o = col / w_tiles, tw = col - (col / w_tiles) * w_tiles;
        return (o * h_tiles + ta) * w_tiles + tw;
    }
    uint32_t output_page() const {
        const uint32_t o = col / w_tiles, tw = col - (col / w_tiles) * w_tiles;
        return (o * chunk_tile_rows() + j) * w_tiles + tw;
    }
    void next() {
        if (++k == column_tiles) {
            k = 0, c = 0, j = 0, ++col;
        } else if (++j == chunk_tiles) {
            j = 0, ++c;
        }
    }
    void advance(uint32_t n) {
        for (uint32_t m = 0; m < n; ++m) {
            next();
        }
    }
};

}  // namespace split
