// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace g6_concat {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

// Valid rows of an output page: only the last tile row of an image can be partial.
FORCE_INLINE uint32_t valid_rows(uint32_t tile_row, uint32_t out_Ht, uint32_t v_last) {
    return tile_row == out_Ht - 1 ? v_last : TILE_ROWS;
}

// Output-geometry cursor (tile column, tile row) advanced one page at a time, so no kernel divides
// per page.
FORCE_INLINE void advance_page(uint32_t& col, uint32_t& tile_row, uint32_t out_Wt, uint32_t out_Ht) {
    if (++col == out_Wt) {
        col = 0;
        if (++tile_row == out_Ht) {
            tile_row = 0;
        }
    }
}

}  // namespace g6_concat
