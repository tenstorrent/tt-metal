// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm: how a cluster's tile-rows are dealt to its C NEOs. The reader, the compute kernel and the writer
// must agree on it exactly (every thread of a kernel takes the same number of entries from a STRIDED ring), so
// it is written once, here.
//
// The cluster owns `num_rows` tile-rows starting at `row_start`. Round k gives NEO t local row k*C + t; a NEO
// left without a real row in the last round recomputes NEO 0's row of that round (the writer stores it again
// with the same bytes), so every NEO runs `rounds` rows.

#pragma once

#include <cstdint>

namespace layer_norm_rows {

// Rows every NEO of the cluster runs, the padded last round included.
constexpr inline uint32_t rounds(uint32_t num_rows, uint32_t C) { return (num_rows + C - 1) / C; }

// The tensor tile-row NEO `neo` handles in round `round`.
constexpr inline uint32_t row_of(uint32_t row_start, uint32_t num_rows, uint32_t C, uint32_t neo, uint32_t round) {
    const uint32_t local = round * C + neo;
    return row_start + (local < num_rows ? local : round * C);
}

}  // namespace layer_norm_rows
