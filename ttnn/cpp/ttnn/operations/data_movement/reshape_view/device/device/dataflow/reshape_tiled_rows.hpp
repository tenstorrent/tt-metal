// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// A tiled reshape of [A, in_rows, C] to [B, out_rows, C] with A * in_rows == B * out_rows: the last dim stays,
// so each output row is one input row, and the rows keep their order. Both kernels walk an output tile's rows in
// runs that stay within one input tile and within one 16-row half of both tiles, so that each run is one
// contiguous span per face column of 32x32 tiles with 16x16 faces.
namespace ttnn::prim::reshape_rows {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

struct Geometry {
    uint32_t in_rows;        // input dim -2
    uint32_t out_rows;       // output dim -2
    uint32_t c_tiles;        // tiles along the last dim
    uint32_t in_row_tiles;   // tiles along input dim -2
    uint32_t out_row_tiles;  // tiles along output dim -2
};

inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Calls f(input_tile, input_row, output_row, num_rows) for each run of output tile out_tile, in row order. The
// tile's rows past out_rows are padding and get no run.
template <typename F>
inline void for_each_run(uint32_t out_tile, const Geometry& g, F&& f) {
    const uint32_t tiles_per_slab = g.out_row_tiles * g.c_tiles;
    const uint32_t b = out_tile / tiles_per_slab;
    const uint32_t in_slab = out_tile - b * tiles_per_slab;
    const uint32_t row_tile = in_slab / g.c_tiles;
    const uint32_t c_tile = in_slab - row_tile * g.c_tiles;

    uint32_t r = row_tile * TILE_ROWS;
    if (r >= g.out_rows) {
        return;
    }
    // Position of the tile's first row among all rows, and the input row that holds it.
    const uint32_t flat = b * g.out_rows + r;
    uint32_t a = flat / g.in_rows;
    uint32_t w = flat - a * g.in_rows;

    uint32_t i = 0;
    while (i < TILE_ROWS && r < g.out_rows) {
        const uint32_t j = w % TILE_ROWS;
        uint32_t n = FACE_ROWS - (i % FACE_ROWS);
        n = min_u32(n, FACE_ROWS - (j % FACE_ROWS));
        n = min_u32(n, g.in_rows - w);
        n = min_u32(n, g.out_rows - r);
        const uint32_t in_tile = (a * g.in_row_tiles + w / TILE_ROWS) * g.c_tiles + c_tile;
        f(in_tile, j, i, n);
        i += n;
        r += n;
        w += n;
        if (w == g.in_rows) {
            w = 0;
            ++a;
        }
    }
}

// The cores walk the output tiles column by column within each slab (b, c_tile, row_tile), so that consecutive output
// tiles share the input tile at their boundary and each kernel keeps it instead of reading it again. Maps that order to
// the output tile's page index.
inline uint32_t output_tile_at(uint32_t k, const Geometry& g) {
    const uint32_t tiles_per_slab = g.out_row_tiles * g.c_tiles;
    const uint32_t b = k / tiles_per_slab;
    const uint32_t in_slab = k - b * tiles_per_slab;
    const uint32_t c_tile = in_slab / g.out_row_tiles;
    const uint32_t row_tile = in_slab - c_tile * g.out_row_tiles;
    return (b * g.out_row_tiles + row_tile) * g.c_tiles + c_tile;
}

// Element offset of row `row` of face column `face_col` in a 32x32 tile of 16x16 faces.
inline uint32_t row_offset(uint32_t row, uint32_t face_col) {
    return (((row / FACE_ROWS) * 2 + face_col) * FACE_ROWS * FACE_ROWS) + ((row % FACE_ROWS) * FACE_ROWS);
}

}  // namespace ttnn::prim::reshape_rows
