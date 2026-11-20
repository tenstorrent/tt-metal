// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing (ttnn.bringup.mhc_pre_xing): shared work walk + tile layouts (reader, writer and compute).
//
// A core owns the contiguous unit range [start, start + count). With streams, a unit is one y column tile
// (token tile-row r, column tile c), r-major, Ct = C / 32 units per row; without streams a unit is one token
// tile-row (Ct = 1). A segment is the run of units inside one row. The three kernels walk the SAME segments.
//
// Layouts (fp32 words of one 32x32 tile in L1):
//   row-major        element (r, c) -> rc_index(r, c) (mhc_pre_layout.hpp)
//   coefficient-major ("SoA", the SFPU view of a DEST tile): slot k, lane l -> 64 (k >> 1) + 2 l + (k & 1)
//                    (= mhc_layout::slot_index(k, l)); lane l = token row l of the tile-row. Every per-token op of
//                    the coefficient stage is lane-wise.
//   pre-block ("PB") tile for the y-mix: slot q = 8 i + b, lane l holds pre_i of token row 4 b + (l >> 3). DEST
//                    vector v (0..31) of a row-major data tile covers the token rows 16 (v >> 4) + 4 ((v >> 1) & 3)
//                    + (l >> 3), so data vector v of stream i is scaled by PB slot 8 i + 4 (v >> 4) + ((v >> 1) & 3).

#pragma once

#include <cstdint>

namespace mhc_xing {

struct Segment {
    uint32_t row;   // token tile-row
    uint32_t col0;  // first unit column of the segment
    uint32_t cols;  // units in the segment
};

struct SegmentWalker {
    uint32_t next_unit;
    uint32_t remaining;
    uint32_t cols_per_row;

    SegmentWalker(uint32_t start, uint32_t count, uint32_t ct) : next_unit(start), remaining(count), cols_per_row(ct) {}

    bool done() const { return remaining == 0; }

    Segment next() {
        Segment s;
        s.row = next_unit / cols_per_row;
        s.col0 = next_unit % cols_per_row;
        const uint32_t left = cols_per_row - s.col0;
        s.cols = remaining < left ? remaining : left;
        next_unit += s.cols;
        remaining -= s.cols;
        return s;
    }
};

constexpr uint32_t pb_slot(uint32_t stream, uint32_t v) { return 8 * stream + 4 * (v >> 4) + ((v >> 1) & 3); }

}  // namespace mhc_xing
