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

// A core's work as two unit segments A = [start_a, start_a + count_a) and B (load balance, split_mode 2:
// e.g. a slice of the source-bearing region and a slice of the fill-only region). Each segment is walked
// by its own RotatedCursor; the two are interleaved in proportion (Bresenham: take A while
// taken_a * count_b <= taken_b * count_a) so every block mixes reads and fills. count_b = 0 is exactly
// the single-segment RotatedCursor walk. `w` / `t` mirror the current segment's cursor.
template <typename Walker>
struct MixCursor {
    RotatedCursor<Walker> a, b;
    uint32_t count_a, count_b, taken_a = 0, taken_b = 0;
    bool on_a = true;
    Walker w;
    uint32_t t;

    MixCursor(
        uint32_t start_a, uint32_t count_a_, uint32_t rot_a, uint32_t start_b, uint32_t count_b_, uint32_t rot_b) :
        a(start_a, count_a_, rot_a),
        b(start_b, count_b_, rot_b),
        count_a(count_a_),
        count_b(count_b_),
        w(start_a + rot_a),
        t(start_a + rot_a) {
        select();
    }

    void select() {
        on_a = taken_a < count_a && (taken_b >= count_b || taken_a * count_b <= taken_b * count_a);
        const RotatedCursor<Walker>& c = on_a ? a : b;
        w = c.w;
        t = c.t;
    }

    void advance() {
        if (on_a) {
            ++taken_a;
            a.advance();
        } else {
            ++taken_b;
            b.advance();
        }
        select();
    }
};

// The cursor the kernels walk: MixCursor when the host split gives a core two segments (split_mode 2,
// CT flag), else the single-segment RotatedCursor (segment B arguments ignored; always count 0 there).
template <typename Walker, bool two_segments>
struct SegmentCursor : RotatedCursor<Walker> {
    SegmentCursor(uint32_t start_a, uint32_t count_a, uint32_t rot_a, uint32_t, uint32_t, uint32_t) :
        RotatedCursor<Walker>(start_a, count_a, rot_a) {}
};
template <typename Walker>
struct SegmentCursor<Walker, true> : MixCursor<Walker> {
    using MixCursor<Walker>::MixCursor;
};

}  // namespace g6_pad
