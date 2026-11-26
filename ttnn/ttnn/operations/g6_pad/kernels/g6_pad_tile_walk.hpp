// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Regime N output-tile walk shared by the reader and the writer, so both classify every tile of a
// block identically (fill-only / input-backed) from the same compile-time geometry.

#pragma once

#include <stdint.h>

// Output tile w = ((l0 * out_lead1 + l1) * out_row_tiles + tr) * out_col_tiles + tc; leading dims
// are padded to two on the host. One decomposition at init, then incremental counters.
template <
    uint32_t out_lead1,
    uint32_t in_lead0,
    uint32_t in_lead1,
    uint32_t out_row_tiles,
    uint32_t out_col_tiles,
    uint32_t in_row_tiles,
    uint32_t in_col_tiles>
struct OutputTileWalker {
    uint32_t l0;
    uint32_t l1;
    uint32_t tr;
    uint32_t tc;
    uint32_t in_slice_base;  // s_in * in_row_tiles * in_col_tiles; valid when slice_exists
    bool slice_exists;

    void init(uint32_t w) {
        tc = w % out_col_tiles;
        const uint32_t rows = w / out_col_tiles;
        tr = rows % out_row_tiles;
        const uint32_t s = rows / out_row_tiles;
        l1 = s % out_lead1;
        l0 = s / out_lead1;
        refresh_slice();
    }

    void refresh_slice() {
        slice_exists = l0 < in_lead0 && l1 < in_lead1;
        in_slice_base = (l0 * in_lead1 + l1) * (in_row_tiles * in_col_tiles);
    }

    void advance() {
        if (++tc < out_col_tiles) {
            return;
        }
        tc = 0;
        if (++tr < out_row_tiles) {
            return;
        }
        tr = 0;
        if (++l1 == out_lead1) {
            l1 = 0;
            ++l0;
        }
        refresh_slice();
    }

    bool has_input() const { return slice_exists && tr < in_row_tiles && tc < in_col_tiles; }

    uint32_t in_page() const { return in_slice_base + tr * in_col_tiles + tc; }

    bool last_in_row() const { return tr == in_row_tiles - 1; }

    bool last_in_col() const { return tc == in_col_tiles - 1; }
};
