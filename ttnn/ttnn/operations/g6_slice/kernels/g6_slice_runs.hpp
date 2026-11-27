// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Byte runs inside one 32x32 tile page for g6_slice's row_runs / col_realign regimes.
//
// A tile page is four 16x16 faces, face-major: face 0 = rows 0-15 cols 0-15, face 1 = rows 0-15
// cols 16-31, face 2 = rows 16-31 cols 0-15, face 3 = rows 16-31 cols 16-31. A row band inside one
// face row (rows [r0, r1) with both ends in the same 16-row half) is one contiguous run per face, and
// runs that touch end to end on both sides are merged, so a whole tile is a single run.
//
// Run lists depend only on the row offset `off`, the row count of the tile row and the column halves
// moved, so each kernel builds them once per row class and replays them for every page.

#include <stdint.h>

namespace g6_slice {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t BOTH_HALVES = 0b11;

struct PageRun {
    uint16_t src;  // byte offset in the source page
    uint16_t dst;  // byte offset in the destination page
    uint16_t len;  // bytes
};

// <= 4 row pieces x 2 column halves.
constexpr uint32_t MAX_RUNS = 8;

// Trivially constructible so kernels can hold run lists in static storage; build with `RunList{}`.
struct RunList {
    PageRun runs[MAX_RUNS];
    uint32_t count;

    void add(uint32_t src, uint32_t dst, uint32_t len) {
        if (count > 0) {
            PageRun& last = runs[count - 1];
            if (last.src + last.len == src && last.dst + last.len == dst) {
                last.len += len;
                return;
            }
        }
        runs[count++] = {static_cast<uint16_t>(src), static_cast<uint16_t>(dst), static_cast<uint16_t>(len)};
    }
};

// Byte offset of row `row`, column half `half` of a tile.
template <uint32_t face_row_bytes>
constexpr uint32_t face_row_offset(uint32_t row, uint32_t half) {
    constexpr uint32_t face_bytes = face_row_bytes * FACE_ROWS;
    return ((row / FACE_ROWS) * 2 + half) * face_bytes + (row % FACE_ROWS) * face_row_bytes;
}

// Rows [r0, r1) of the selected column halves, read to the same position in the staging page.
// DRAM reads need L1 dst == src (mod 64); keeping the natural in-tile offset satisfies it.
template <uint32_t face_row_bytes>
RunList band_runs(uint32_t r0, uint32_t r1, uint32_t half_mask) {
    RunList list{};
    while (r0 < r1) {
        const uint32_t face_end = (r0 / FACE_ROWS + 1) * FACE_ROWS;
        const uint32_t piece_end = r1 < face_end ? r1 : face_end;
        for (uint32_t half = 0; half < 2; ++half) {
            if (half_mask & (1u << half)) {
                const uint32_t offset = face_row_offset<face_row_bytes>(r0, half);
                list.add(offset, offset, (piece_end - r0) * face_row_bytes);
            }
        }
        r0 = piece_end;
    }
    return list;
}

// Output rows [0, rows) of a page from staging rows (i + off) mod 32, whole tile width. Pieces break
// wherever the source or destination crosses a face boundary or the source wraps past row 31.
template <uint32_t face_row_bytes>
RunList rotate_runs(uint32_t off, uint32_t rows) {
    RunList list{};
    uint32_t i = 0;
    while (i < rows) {
        const uint32_t s = (i + off) % TILE_ROWS;
        const uint32_t dst_end = (i / FACE_ROWS + 1) * FACE_ROWS;
        const uint32_t src_end = i + ((s / FACE_ROWS + 1) * FACE_ROWS - s);
        uint32_t piece_end = rows;
        piece_end = dst_end < piece_end ? dst_end : piece_end;
        piece_end = src_end < piece_end ? src_end : piece_end;
        for (uint32_t half = 0; half < 2; ++half) {
            list.add(
                face_row_offset<face_row_bytes>(s, half),
                face_row_offset<face_row_bytes>(i, half),
                (piece_end - i) * face_row_bytes);
        }
        i = piece_end;
    }
    return list;
}

}  // namespace g6_slice
