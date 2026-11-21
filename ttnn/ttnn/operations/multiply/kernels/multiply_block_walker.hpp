// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The block decomposition of the `row_bcast_strips` regime: the single source of truth that the reader, the
// compute kernel and the writer all walk, so the three agree on every block.
//
// Units are column strips (n, g, c): `rows(g)` tiles of output column c in row group g of output image n,
// numbered u = (n * G + g) * Wt + c (c fastest). A core owns the run [start_unit, start_unit + num_units).
// The walker cuts that run into blocks at every (n, g) boundary and every `block_cols` columns; one block is
// `rows x cols` output tiles, inside one image and one row group, with `cols <= block_cols`.
//
// Row group g of G spans tile rows [floor(g * Ht / G), floor((g + 1) * Ht / G)).

#pragma once

#include <cstdint>

namespace multiply {

struct Block {
    uint32_t n;          // output image (folded leading dims)
    uint32_t row_begin;  // first tile row of the block in the image
    uint32_t rows;       // tile rows in the block (the row group's height)
    uint32_t col_begin;  // first tile column of the block
    uint32_t cols;       // tile columns in the block, <= block_cols
};

class BlockWalker {
public:
    BlockWalker(uint32_t start_unit, uint32_t num_units, uint32_t G, uint32_t Ht, uint32_t Wt, uint32_t block_cols) :
        u_(start_unit), end_(start_unit + num_units), G_(G), Ht_(Ht), Wt_(Wt), block_cols_(block_cols) {}

    // Writes the next block into `b`; false when the run is exhausted.
    bool next(Block& b) {
        if (u_ >= end_) {
            return false;
        }
        const uint32_t ng = u_ / Wt_;
        const uint32_t c = u_ - ng * Wt_;
        const uint32_t g = ng % G_;
        uint32_t cols = Wt_ - c;
        if (cols > block_cols_) {
            cols = block_cols_;
        }
        if (cols > end_ - u_) {
            cols = end_ - u_;
        }
        b.n = ng / G_;
        b.row_begin = g * Ht_ / G_;
        b.rows = (g + 1) * Ht_ / G_ - b.row_begin;
        b.col_begin = c;
        b.cols = cols;
        u_ += cols;
        return true;
    }

private:
    uint32_t u_;
    uint32_t end_;
    uint32_t G_;
    uint32_t Ht_;
    uint32_t Wt_;
    uint32_t block_cols_;
};

}  // namespace multiply
