// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::ccl::snake_ring {

// Shared by host route construction and device dataflow kernels. The
// dimensions and orientation are compile-time constants in device callers so
// division and modulo do not become runtime dataflow-core operations.
//
// Two families of Hamiltonian cycle over a 2D mesh, both requiring one even
// extent and both using nearest-neighbour edges only *except* where noted:
//
//   Row / Column   boustrophedon. Lanes are traversed in alternating
//                  directions, so the last device sits at the far end of the
//                  closing axis and the closing edge spans that whole axis --
//                  a direct hop only on a torus, or across an extent-2 axis.
//   CombRow /      spine along one full lane, then a boustrophedon over the
//   CombColumn     remaining lanes running back towards the spine's origin.
//                  Every edge including the closing one is a nearest
//                  neighbour, so these close on a plain (non-torus) mesh.
//
// The comb costs nothing the boustrophedon does not: same parity precondition
// (CombRow needs an even row count, exactly as Row does), same cycle length,
// same one-hop edges. It only trades which physical links the cycle rides.
enum class Orientation : uint32_t { Row = 0, Column = 1, CombRow = 2, CombColumn = 3 };

constexpr bool is_comb(Orientation orientation) {
    return orientation == Orientation::CombRow || orientation == Orientation::CombColumn;
}

// Lanes are rows for Row/CombRow and columns for Column/CombColumn. Both
// families need this count even, so callers validate it the same way.
constexpr uint32_t lane_count(uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    return (orientation == Orientation::Row || orientation == Orientation::CombRow) ? num_rows : num_cols;
}

// This alternating-major-lane permutation is an involution. For a row snake,
// a lane is a physical row and lane_width is the number of columns. For a
// column snake, a lane is a physical column and lane_width is the number of
// rows.
constexpr uint32_t permute_odd_lanes(uint32_t linear_index, uint32_t lane_width) {
    if (lane_width == 0) {
        return 0;
    }
    const uint32_t lane = linear_index / lane_width;
    const uint32_t offset = linear_index % lane_width;
    return lane * lane_width + ((lane & 1U) == 0 ? offset : lane_width - 1 - offset);
}

////////////////////////////////////////////////////////////////
// Comb cycle, stated for CombRow on an R x C mesh with R even. CombColumn is
// the transpose of the same walk (spine across row 0, lanes are columns), so
// both share this implementation with the axes swapped by the callers below.
//
// Ranks [0, R) run the spine straight down column 0: rank r is at (r, 0).
// The remaining R*(C-1) ranks zig-zag columns 1..C-1, walking rows from the
// bottom (R-1) back up to row 0. Lane `s` (= R-1-r) runs left-to-right when s
// is even and right-to-left when s is odd, so the final lane (s = R-1, odd
// because R is even) ends at column 1 -- adjacent to the spine's origin, which
// is what closes the cycle without a wrap.
//
// Every division here is by (C-1), a compile-time constant in device callers.
////////////////////////////////////////////////////////////////

// (row, col) -> rank, for a comb whose spine runs down `col 0` of a
// `lane_span` x `teeth_span` grid. `major`/`minor` are (row, col) for CombRow.
constexpr uint32_t comb_index(uint32_t major, uint32_t minor, uint32_t lane_span, uint32_t teeth_span) {
    if (lane_span == 0 || teeth_span < 2) {
        return 0;
    }
    if (minor == 0) {
        return major;  // on the spine
    }
    const uint32_t tooth_width = teeth_span - 1;
    const uint32_t lane = lane_span - 1 - major;
    const uint32_t offset = (lane & 1U) == 0 ? minor - 1 : teeth_span - 1 - minor;
    return lane_span + lane * tooth_width + offset;
}

// rank -> major coordinate (row for CombRow).
constexpr uint32_t comb_major(uint32_t rank, uint32_t lane_span, uint32_t teeth_span) {
    if (lane_span == 0 || teeth_span < 2) {
        return 0;
    }
    if (rank < lane_span) {
        return rank;  // on the spine
    }
    return lane_span - 1 - (rank - lane_span) / (teeth_span - 1);
}

// rank -> minor coordinate (col for CombRow).
constexpr uint32_t comb_minor(uint32_t rank, uint32_t lane_span, uint32_t teeth_span) {
    if (lane_span == 0 || teeth_span < 2) {
        return 0;
    }
    if (rank < lane_span) {
        return 0;  // on the spine
    }
    const uint32_t tooth_width = teeth_span - 1;
    const uint32_t lane = (rank - lane_span) / tooth_width;
    const uint32_t offset = (rank - lane_span) % tooth_width;
    return (lane & 1U) == 0 ? offset + 1 : teeth_span - 1 - offset;
}

constexpr uint32_t index_from_coordinate(
    uint32_t row, uint32_t col, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    switch (orientation) {
        case Orientation::Row: return permute_odd_lanes(row * num_cols + col, num_cols);
        case Orientation::Column: return permute_odd_lanes(col * num_rows + row, num_rows);
        case Orientation::CombRow: return comb_index(row, col, num_rows, num_cols);
        case Orientation::CombColumn: return comb_index(col, row, num_cols, num_rows);
    }
    return 0;
}

constexpr uint32_t coordinate_row(uint32_t ring_index, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    if (num_rows == 0 || num_cols == 0) {
        return 0;
    }
    switch (orientation) {
        case Orientation::Row: return ring_index / num_cols;
        case Orientation::Column: return permute_odd_lanes(ring_index, num_rows) % num_rows;
        case Orientation::CombRow: return comb_major(ring_index, num_rows, num_cols);
        case Orientation::CombColumn: return comb_minor(ring_index, num_cols, num_rows);
    }
    return 0;
}

constexpr uint32_t coordinate_col(uint32_t ring_index, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    if (num_rows == 0 || num_cols == 0) {
        return 0;
    }
    switch (orientation) {
        case Orientation::Row: return permute_odd_lanes(ring_index, num_cols) % num_cols;
        case Orientation::Column: return ring_index / num_rows;
        case Orientation::CombRow: return comb_minor(ring_index, num_rows, num_cols);
        case Orientation::CombColumn: return comb_major(ring_index, num_cols, num_rows);
    }
    return 0;
}

constexpr uint32_t row_major_index(uint32_t ring_index, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    const uint32_t row = coordinate_row(ring_index, num_rows, num_cols, orientation);
    const uint32_t col = coordinate_col(ring_index, num_rows, num_cols, orientation);
    return row * num_cols + col;
}

}  // namespace ttnn::ccl::snake_ring
