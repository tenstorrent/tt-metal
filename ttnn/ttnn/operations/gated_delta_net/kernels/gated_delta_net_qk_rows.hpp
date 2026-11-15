// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// R-QKCORE window gather, shared by the q / k core's reader (NCRISC) and writer (BRISC), which split the rows.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace gdn {

// History taps of rows [p_lo, p_hi) of one column: tap j < HISTORY row p = conv_state row HISTORY p + j, from the
// column's cs_tiles staged conv_state tiles into its zeroed tap tiles at col_win. Every row is two face-row segments
// inside this core: one stateful command (own NoC coordinates, segment size), then one address pair per segment.
// Completed by the caller's read barrier.
template <uint32_t TILE_BYTES, uint32_t HISTORY>
FORCE_INLINE void copy_history_rows(uint32_t cs_col, uint32_t col_win, uint32_t p_lo, uint32_t p_hi) {
    constexpr uint32_t TILE_ROWS = 32;
    constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;
    constexpr uint32_t SEG_BYTES = FACE_BYTES / 16;
    auto row_offset = [](uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; };
    noc_async_read_one_packet_set_state(get_noc_addr(cs_col), SEG_BYTES);
    for (uint32_t p = p_lo; p < p_hi; ++p) {
        for (uint32_t tap = 0; tap < HISTORY; ++tap) {
            const uint32_t r = HISTORY * p + tap;
            const uint32_t src = cs_col + (r / TILE_ROWS) * TILE_BYTES + row_offset(r % TILE_ROWS);
            const uint32_t dst = col_win + tap * TILE_BYTES + row_offset(p);
            noc_async_read_one_packet_with_state(src, dst);
            noc_async_read_one_packet_with_state(src + FACE_BYTES, dst + FACE_BYTES);
        }
    }
}

// Rows [0, split) are the reader's, [split, B) the writer's.
FORCE_INLINE uint32_t history_split(uint32_t batch) { return (batch + 1) / 2; }

}  // namespace gdn
