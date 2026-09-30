// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — shared segment / block derivation (included by reader, compute and writer).
//
// A core owns the contiguous flattened unit range [start_unit, start_unit + num_units), one unit being
// (token-tile row r, column tile c), r-major. A *segment* is the maximal run of those units inside one
// token-tile row; a segment is walked in blocks of `block_col_tiles` columns, the last one possibly
// ragged (`valid_col_tiles < block_col_tiles`). All three kernels iterate through THIS derivation so
// they cannot disagree on segment / block boundaries (a disagreement is a CB count mismatch -> hang).

#pragma once

#include <cstdint>

namespace mhc_post {

struct Segment {
    uint32_t row;        // token-tile row r
    uint32_t col0;       // first column tile c of the segment
    uint32_t col_tiles;  // number of column tiles in the segment
};

// Iterates the segments of one core's unit range.
struct SegmentWalker {
    uint32_t next_unit;
    uint32_t remaining;
    uint32_t col_tiles_per_row;  // Ct

    SegmentWalker(uint32_t start_unit, uint32_t num_units, uint32_t ct) :
        next_unit(start_unit), remaining(num_units), col_tiles_per_row(ct) {}

    bool done() const { return remaining == 0; }

    Segment next() {
        Segment s;
        s.row = next_unit / col_tiles_per_row;
        s.col0 = next_unit % col_tiles_per_row;
        const uint32_t row_left = col_tiles_per_row - s.col0;
        s.col_tiles = remaining < row_left ? remaining : row_left;
        next_unit += s.col_tiles;
        remaining -= s.col_tiles;
        return s;
    }
};

// Number of blocks of `block_col_tiles` covering `seg_col_tiles` (ceil).
inline uint32_t num_blocks(uint32_t seg_col_tiles, uint32_t block_col_tiles) {
    return (seg_col_tiles + block_col_tiles - 1) / block_col_tiles;
}

// Valid (runtime) column extent of block `block_idx` of a segment.
inline uint32_t block_valid_col_tiles(uint32_t seg_col_tiles, uint32_t block_col_tiles, uint32_t block_idx) {
    const uint32_t done = block_idx * block_col_tiles;
    const uint32_t left = seg_col_tiles - done;
    return left < block_col_tiles ? left : block_col_tiles;
}

// ---- Expanded coefficient layout (cb_coef_bcast; reader writes it, compute reads it) ----
// Output stream j has n+1 terms: t = 0 is post_j, t = 1+i is comb[i][j] (comb applied transposed).
// Two terms share one fp32 tile: term t occupies HALF (t % 2) of tile j*P + t/2, with
// P = coef_tiles_per_stream = ceil((n+1)/2) (host CT arg). Half 0 = faces 0/2 (tile columns 0-15), half 1 =
// faces 1/3 (columns 16-31); every element of tile row rho in a half holds the coefficient of token row rho.
// A half carries everything a data face needs (the coefficient is constant along columns), so the SFPU
// reads data faces 2h and 2h+1 against coefficient face 2h + (t % 2).
constexpr uint32_t coef_tile_in_stream(uint32_t t) { return t / 2; }
constexpr uint32_t coef_half(uint32_t t) { return t % 2; }

}  // namespace mhc_post
