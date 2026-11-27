// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Byte-run geometry of g6_slice, shared by reader and writer.
//
// A 32x32 tile is four 16x16 faces stored face by face (0: rows 0-15 / cols 0-15, 1: rows 0-15 /
// cols 16-31, 2: rows 16-31 / cols 0-15, 3: rows 16-31 / cols 16-31), row-major inside a face.
// A band of whole rows is therefore at most 4 contiguous runs of whole face-rows; contiguous
// pieces are merged, so a whole tile is one run.
//
// The staging page in cb_slice_pages holds every input row at its natural in-tile position
// (DRAM reads need dst == src mod 64), so staging row s holds output row (s - off) mod 32. The
// writer applies the rotation (DRAM writes need only src == dst mod 16; face-rows are 32 B or 64 B).
//
// Run lists depend only on (off, rows of the output tile), so each kernel builds them once per
// row class (full tile row, last tile row of an image) and replays them for every page.

#pragma once

#include <stdint.h>

namespace g6_slice {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t MAX_RUNS = 8;  // 2 row bands (or 4 rotated pieces) x 2 face halves x 2 face columns, merged

constexpr uint32_t COL_HALF_LEFT = 1;   // faces 0 and 2
constexpr uint32_t COL_HALF_RIGHT = 2;  // faces 1 and 3
constexpr uint32_t COL_HALF_BOTH = COL_HALF_LEFT | COL_HALF_RIGHT;

struct ByteRun {
    uint32_t src_offset;  // byte offset in the source page
    uint32_t dst_offset;  // byte offset in the destination page
    uint32_t bytes;
    uint32_t src_tile;  // reader: 0 = upper input tile, 1 = the tile below it
};

struct RunList {
    ByteRun runs[MAX_RUNS];
    uint32_t count = 0;

    void append(uint32_t src_tile, uint32_t src_offset, uint32_t dst_offset, uint32_t bytes) {
        if (count > 0) {
            ByteRun& last = runs[count - 1];
            if (last.src_tile == src_tile && last.src_offset + last.bytes == src_offset &&
                last.dst_offset + last.bytes == dst_offset) {
                last.bytes += bytes;
                return;
            }
        }
        runs[count++] = ByteRun{src_offset, dst_offset, bytes, src_tile};
    }
};

template <uint32_t page_bytes>
struct TileGeometry {
    static_assert(page_bytes % (TILE_DIM * TILE_DIM) == 0, "row runs need 16- or 32-bit elements");
    static constexpr uint32_t face_row_bytes = page_bytes / (4 * FACE_DIM);
    static constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;

    static constexpr uint32_t offset(uint32_t row, uint32_t col_half) {
        return ((row / FACE_DIM) * 2 + col_half) * face_bytes + (row % FACE_DIM) * face_row_bytes;
    }

    // Rows [r0, r1) of a tile, at their natural positions in source and destination.
    static void add_row_band(RunList& list, uint32_t src_tile, uint32_t r0, uint32_t r1, uint32_t col_halves) {
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t a = r0 > half * FACE_DIM ? r0 : half * FACE_DIM;
            const uint32_t b = r1 < (half + 1) * FACE_DIM ? r1 : (half + 1) * FACE_DIM;
            if (a >= b) {
                continue;
            }
            for (uint32_t col_half = 0; col_half < 2; ++col_half) {
                if (col_halves & (1u << col_half)) {
                    const uint32_t at = offset(a, col_half);
                    list.append(src_tile, at, at, (b - a) * face_row_bytes);
                }
            }
        }
    }

    // Reader: the bands of input rows that output rows [0, rv) draw from, at natural positions.
    static RunList fetch_runs(uint32_t off, uint32_t rv, uint32_t col_halves) {
        RunList list;
        const uint32_t upper_rows = (TILE_DIM - off) < rv ? (TILE_DIM - off) : rv;
        add_row_band(list, 0, off, off + upper_rows, col_halves);
        add_row_band(list, 1, 0, rv - upper_rows, col_halves);
        return list;
    }

    // Writer: output rows [0, rv) from staging rows (i + off) mod 32, split where either side
    // crosses a face boundary (16 rows) or the staging rows wrap.
    static RunList store_runs(uint32_t off, uint32_t rv) {
        RunList list;
        uint32_t i = 0;
        while (i < rv) {
            const uint32_t s = (i + off) % TILE_DIM;
            uint32_t rows = rv - i;
            if (FACE_DIM - i % FACE_DIM < rows) {
                rows = FACE_DIM - i % FACE_DIM;
            }
            if (FACE_DIM - s % FACE_DIM < rows) {
                rows = FACE_DIM - s % FACE_DIM;
            }
            for (uint32_t col_half = 0; col_half < 2; ++col_half) {
                list.append(0, offset(s, col_half), offset(i, col_half), rows * face_row_bytes);
            }
            i += rows;
        }
        return list;
    }
};

// Output-page walk (o1, th, tw) with carries tw -> th -> o1; o0 needs no counter. `in_page` is the
// upper input tile of the current output page, kept incrementally.
template <
    uint32_t tensor_in_ht_tiles,
    uint32_t tensor_in_wt_tiles,
    uint32_t in_lead1_extent,
    uint32_t tensor_out_ht_tiles,
    uint32_t tensor_out_wt_tiles,
    uint32_t out_lead1_extent>
struct OutPageWalk {
    uint32_t o1;
    uint32_t th;
    uint32_t tw;
    uint32_t in_page;

    bool last_tile_row() const { return th == tensor_out_ht_tiles - 1; }
    bool last_tile_col() const { return tw == tensor_out_wt_tiles - 1; }

    void advance() {
        ++in_page;
        if (++tw == tensor_out_wt_tiles) {
            tw = 0;
            in_page += tensor_in_wt_tiles - tensor_out_wt_tiles;
            if (++th == tensor_out_ht_tiles) {
                th = 0;
                in_page += (tensor_in_ht_tiles - tensor_out_ht_tiles) * tensor_in_wt_tiles;
                if (++o1 == out_lead1_extent) {
                    o1 = 0;
                    in_page += (in_lead1_extent - out_lead1_extent) * tensor_in_ht_tiles * tensor_in_wt_tiles;
                }
            }
        }
    }
};

}  // namespace g6_slice
