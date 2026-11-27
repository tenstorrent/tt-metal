// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Geometry, block knobs and the block walk shared by every reader and writer of g6_to_layout.
// The walk must be identical in the reader and the writer of a regime: both cut the core's
// tile range into the same blocks, and the CB carries exactly one page per block.

namespace g6_to_layout {

constexpr uint32_t TILE_HW = 32;
constexpr uint32_t FACE_HW = 16;
constexpr uint32_t DATUM_BYTES = 2;  // bfloat16
constexpr uint32_t FACE_ROW_BYTES = FACE_HW * DATUM_BYTES;
constexpr uint32_t FACE_BYTES = FACE_HW * FACE_ROW_BYTES;
constexpr uint32_t TILE_BYTES = 4 * FACE_BYTES;
// R2 slot of one tile: faces 0 and 2 only (faces 1 and 3 hold no live column when W <= 16).
constexpr uint32_t LEFT_SLOT_BYTES = 2 * FACE_BYTES;
// Bytes one tile column contributes to an output row.
constexpr uint32_t TILE_ROW_BYTES = TILE_HW * DATUM_BYTES;

// Compile-time args 0..6 of every kernel; kernel-specific args start at `next_ct_arg`.
struct Geometry {
    static constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(0);
    static constexpr uint32_t image_rows_tiles = get_compile_time_arg_val(1);
    static constexpr uint32_t height = get_compile_time_arg_val(2);
    static constexpr uint32_t width = get_compile_time_arg_val(3);
    static constexpr uint32_t block_rows_tiles = get_compile_time_arg_val(4);
    static constexpr uint32_t block_cols_tiles = get_compile_time_arg_val(5);
    // Rows a tile-row occupies in the row stage: min(32, H).
    static constexpr uint32_t tile_row_stage_rows = get_compile_time_arg_val(6);
    static constexpr uint32_t next_ct_arg = 7;

    static constexpr bool full_row_blocks = block_cols_tiles == tensor_cols_tiles;
    static constexpr uint32_t stage_row_bytes = block_cols_tiles * TILE_ROW_BYTES;
    static constexpr uint32_t stage_tile_row_bytes = tile_row_stage_rows * stage_row_bytes;
};

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Rows of tensor tile-row `g` that hold logical values (each image is tile-padded on its own).
FORCE_INLINE uint32_t live_rows_of(uint32_t g) {
    const uint32_t ht = g % Geometry::image_rows_tiles;
    return min_u32(TILE_HW, Geometry::height - ht * TILE_HW);
}

// Output page of row 0 of tensor tile-row `g`: pages are contiguous across images.
FORCE_INLINE uint32_t first_out_page_of(uint32_t g) {
    const uint32_t image = g / Geometry::image_rows_tiles;
    const uint32_t ht = g - image * Geometry::image_rows_tiles;
    return image * Geometry::height + ht * TILE_HW;
}

// Columns of tile column `c` that hold logical values.
FORCE_INLINE uint32_t live_cols_of(uint32_t c) { return min_u32(TILE_HW, Geometry::width - c * TILE_HW); }

// A rectangle of tiles: `rows` tile-rows from `tile_row`, `cols` tile columns from `col`.
// Ragged extents are runtime values; the CB page stays the nominal block.
struct Block {
    uint32_t tile_row;
    uint32_t rows;
    uint32_t col;
    uint32_t cols;
};

// Cuts the core's flattened tile range [begin, end) into blocks. With full-row blocks a block
// is a run of up to block_rows_tiles whole tile-rows and a partial head/tail row is its own
// block; otherwise every row span is cut into blocks of up to block_cols_tiles columns.
struct BlockWalk {
    uint32_t tile;
    uint32_t end;

    FORCE_INLINE bool next(Block& block) {
        using G = Geometry;
        if (tile >= end) {
            return false;
        }
        const uint32_t g = tile / G::tensor_cols_tiles;
        const uint32_t col = tile - g * G::tensor_cols_tiles;
        const uint32_t span_cols = min_u32(G::tensor_cols_tiles - col, end - tile);
        if (G::full_row_blocks && col == 0 && span_cols == G::tensor_cols_tiles) {
            const uint32_t rows = min_u32(G::block_rows_tiles, (end - tile) / G::tensor_cols_tiles);
            block = Block{g, rows, 0, G::tensor_cols_tiles};
            tile += rows * G::tensor_cols_tiles;
        } else {
            const uint32_t cols = min_u32(G::block_cols_tiles, span_cols);
            block = Block{g, 1, col, cols};
            tile += cols;
        }
        return true;
    }
};

}  // namespace g6_to_layout
