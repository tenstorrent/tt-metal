// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad: shared output-tile classification for the new_storage_tiles reader and writer.
// Both kernels walk the same flat output-tile range with the same predicate (op_design.md,
// "Tile classification"); the host never enumerates tiles.

#pragma once

#include <stdint.h>

namespace g6_pad {

// Leading dims are folded to two (lead0, lead1); rank 2/3 pad with leading 1s on the host.
template <
    uint32_t out_tiles_h,
    uint32_t out_tiles_w,
    uint32_t out_lead1,
    uint32_t in_tiles_h,
    uint32_t in_tiles_w,
    uint32_t in_lead0,
    uint32_t in_lead1>
struct OutTileWalker {
    uint32_t wt = 0, ht = 0, j1 = 0, j0 = 0;

    // Flat output tile t = ((j0 * out_lead1 + j1) * out_tiles_h + ht) * out_tiles_w + wt.
    explicit OutTileWalker(uint32_t t) {
        wt = t % out_tiles_w;
        t /= out_tiles_w;
        ht = t % out_tiles_h;
        t /= out_tiles_h;
        j1 = t % out_lead1;
        j0 = t / out_lead1;
    }

    void advance() {
        if (++wt == out_tiles_w) {
            wt = 0;
            if (++ht == out_tiles_h) {
                ht = 0;
                if (++j1 == out_lead1) {
                    j1 = 0;
                    ++j0;
                }
            }
        }
    }

    bool source_bearing() const { return j0 < in_lead0 && j1 < in_lead1 && ht < in_tiles_h && wt < in_tiles_w; }

    uint32_t source_page() const { return ((j0 * in_lead1 + j1) * in_tiles_h + ht) * in_tiles_w + wt; }

    bool last_input_tile_row() const { return ht == in_tiles_h - 1; }
    bool last_input_tile_col() const { return wt == in_tiles_w - 1; }
};

}  // namespace g6_pad
