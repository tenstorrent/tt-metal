// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad: shared output-unit classification for the new_storage reader and writer (tiles: unit = output
// tile; rows: unit = (output row, row chunk)).
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

    // new_storage_rows: the walker's "tile row" is one row and its "tile column" is one row chunk, so a
    // unit is (row, chunk); these give the input / output row (= page) index of the current unit.
    uint32_t source_row() const { return (j0 * in_lead1 + j1) * in_tiles_h + ht; }
    uint32_t out_row() const { return (j0 * out_lead1 + j1) * out_tiles_h + ht; }

    bool last_input_tile_row() const { return ht == in_tiles_h - 1; }
    bool last_input_tile_col() const { return wt == in_tiles_w - 1; }
};

// A core's unit range [start, start + count) walked from start + rot and wrapped to start (bank_stagger:
// the host picks rot per core so concurrent cores begin on different DRAM banks). rot = 0 is the plain
// in-order walk. Reader and writer construct it with the same (start, count, rot), so slot k of every
// block holds the same unit on both sides. `t` is the flat unit index of the current unit.
template <typename Walker>
struct RotatedCursor {
    Walker w;
    uint32_t t, start, end;

    RotatedCursor(uint32_t start_, uint32_t count, uint32_t rot) :
        w(start_ + rot), t(start_ + rot), start(start_), end(start_ + count) {}

    void advance() {
        if (++t == end) {
            t = start;
            w = Walker(start);
        } else {
            w.advance();
        }
    }
};

}  // namespace g6_pad
