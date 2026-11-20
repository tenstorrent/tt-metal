// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The tiled reshape that keeps the last dim: [C_in, H_in, W] -> [C_out, H_out, W] with C_in * H_in ==
// C_out * H_out. Every output row is one input row of the same width, so an output tile's rows come
// from a few runs of consecutive rows of single input tiles, in the same tile column. Both kernels
// include it, so the reader and the writer walk the same runs.
namespace ttnn::prim::detail::reshape_rows {

constexpr uint32_t kTileRows = 32;

// The row counts of one call; they are runtime args, so one program serves every row count.
struct Rows {
    uint32_t in_rows;        // H_in, logical
    uint32_t in_tile_rows;   // tiles per input slice along H: div_up(H_in, 32)
    uint32_t out_rows;       // H_out, logical
    uint32_t out_tile_rows;  // tiles per output slice along H: div_up(H_out, 32)
    uint32_t tiles_per_row;  // tiles along W, the same for input and output
};

// A run of `count` rows: input tile `in_page` from row `in_row`, to output rows from `out_row`.
struct Run {
    uint32_t in_page;
    uint32_t in_row;
    uint32_t out_row;
    uint32_t count;
};

// The logical rows of output tile `out_page`; the rest of its 32 rows are tile padding.
inline uint32_t rows_in_tile(const Rows& r, const uint32_t out_page) {
    const uint32_t tile_h = (out_page / r.tiles_per_row) % r.out_tile_rows;
    const uint32_t rows_left = r.out_rows - (tile_h * kTileRows);
    return rows_left < kTileRows ? rows_left : kTileRows;
}

// Calls `emit(Run)` for each run of output tile `out_page`, in output row order. An output row past
// H_out is tile padding and has no run.
template <typename Emit>
inline void for_each_run(const Rows& r, const uint32_t out_page, Emit&& emit) {
    const uint32_t tiles_per_slice = r.out_tile_rows * r.tiles_per_row;
    const uint32_t c_out = out_page / tiles_per_slice;
    const uint32_t in_slice = out_page - (c_out * tiles_per_slice);
    const uint32_t tile_h = in_slice / r.tiles_per_row;
    const uint32_t tile_w = in_slice - (tile_h * r.tiles_per_row);

    const uint32_t first_row = tile_h * kTileRows;
    const uint32_t rows_left = r.out_rows - first_row;
    const uint32_t rows = rows_left < kTileRows ? rows_left : kTileRows;

    // The same flat row in the input: slice c_in, row h_in.
    const uint32_t flat = (c_out * r.out_rows) + first_row;
    uint32_t c_in = flat / r.in_rows;
    uint32_t h_in = flat - (c_in * r.in_rows);

    uint32_t out_row = 0;
    while (out_row < rows) {
        const uint32_t in_row = h_in % kTileRows;
        uint32_t count = rows - out_row;
        if (kTileRows - in_row < count) {
            count = kTileRows - in_row;
        }
        if (r.in_rows - h_in < count) {
            count = r.in_rows - h_in;
        }
        const uint32_t in_page = (((c_in * r.in_tile_rows) + (h_in / kTileRows)) * r.tiles_per_row) + tile_w;
        emit(Run{in_page, in_row, out_row, count});
        out_row += count;
        h_in += count;
        if (h_in == r.in_rows) {
            h_in = 0;
            ++c_in;
        }
    }
}

}  // namespace ttnn::prim::detail::reshape_rows
