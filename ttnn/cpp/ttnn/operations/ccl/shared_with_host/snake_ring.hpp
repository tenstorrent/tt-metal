// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::ccl::snake_ring {

// Shared by host route construction and device dataflow kernels. The
// dimensions and orientation are compile-time constants in device callers so
// division and modulo do not become runtime dataflow-core operations.
enum class Orientation : uint32_t { Row = 0, Column = 1 };

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

constexpr uint32_t index_from_coordinate(
    uint32_t row, uint32_t col, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    return orientation == Orientation::Row ? permute_odd_lanes(row * num_cols + col, num_cols)
                                           : permute_odd_lanes(col * num_rows + row, num_rows);
}

constexpr uint32_t coordinate_row(uint32_t ring_index, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    if (num_rows == 0 || num_cols == 0) {
        return 0;
    }
    return orientation == Orientation::Row ? ring_index / num_cols : permute_odd_lanes(ring_index, num_rows) % num_rows;
}

constexpr uint32_t coordinate_col(uint32_t ring_index, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    if (num_rows == 0 || num_cols == 0) {
        return 0;
    }
    return orientation == Orientation::Row ? permute_odd_lanes(ring_index, num_cols) % num_cols : ring_index / num_rows;
}

constexpr uint32_t row_major_index(uint32_t ring_index, uint32_t num_rows, uint32_t num_cols, Orientation orientation) {
    const uint32_t row = coordinate_row(ring_index, num_rows, num_cols, orientation);
    const uint32_t col = coordinate_col(ring_index, num_rows, num_cols, orientation);
    return row * num_cols + col;
}

}  // namespace ttnn::ccl::snake_ring
