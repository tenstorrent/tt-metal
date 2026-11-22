// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace kda_relay {

// This correction core's head and place among the head's cores: head h's cores are a column of the compute grid,
// x = h / heads_per_column, rows first_row + (h % heads_per_column) * cores_per_head + place (the summary's value
// blocks of a head take a grid column the same way, so the terms it writes stay in the column; one multicast of the
// head's W reaches them all).
struct CorrectionCore {
    uint32_t head;
    uint32_t place;
};
template <uint32_t heads_per_column, uint32_t first_row, uint32_t cores_per_head>
inline CorrectionCore correction_core() {
    const uint32_t row = get_absolute_logical_y() - first_row;
    return {get_absolute_logical_x() * heads_per_column + row / cores_per_head, row % cores_per_head};
}

}  // namespace kda_relay
