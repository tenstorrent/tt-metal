// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Byte runs of one 32x32 tile page for the row-run regimes of g6_slice.
//
// A tile page is four 16x16 faces, face-major: face 0 = rows 0-15 / cols 0-15, face 1 = rows 0-15 /
// cols 16-31, face 2 = rows 16-31 / cols 0-15, face 3 = rows 16-31 / cols 16-31. A face row is
// `face_row_bytes` (page_bytes / 64) and a face 16 of them. A band of tile rows therefore maps to at
// most two face rows x two column halves; adjacent pieces that are contiguous on both sides merge,
// so a whole tile is a single run.
//
// Run lists depend only on (row offset, rows in the tile) and are built once per kernel per row class.

#include <stdint.h>

namespace g6_slice {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t TILE_COLS = 32;
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t BOTH_HALVES = 0b11;

struct ByteRun {
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t size;
};

template <uint32_t capacity>
struct RunList {
    ByteRun runs[capacity];
    uint32_t count = 0;

    void append(uint32_t src_offset, uint32_t dst_offset, uint32_t size) {
        if (count > 0) {
            ByteRun& last = runs[count - 1];
            if (last.src_offset + last.size == src_offset && last.dst_offset + last.size == dst_offset) {
                last.size += size;
                return;
            }
        }
        runs[count++] = ByteRun{src_offset, dst_offset, size};
    }
};

// Byte offset of tile row `row` in column half `half`.
template <uint32_t face_row_bytes>
constexpr uint32_t tile_row_offset(uint32_t row, uint32_t half) {
    constexpr uint32_t face_bytes = FACE_ROWS * face_row_bytes;
    return ((row / FACE_ROWS) * 2 + half) * face_bytes + (row % FACE_ROWS) * face_row_bytes;
}

// Rows [row_begin, row_end) of a tile, restricted to the column halves in `half_mask`, at their
// natural positions on both sides. Reads keep the in-tile offset so the L1 destination and the DRAM
// source agree modulo the DRAM read alignment.
template <uint32_t face_row_bytes>
RunList<4> band_runs(uint32_t row_begin, uint32_t row_end, uint32_t half_mask) {
    RunList<4> list;
    for (uint32_t face_row = 0; face_row < 2; ++face_row) {
        const uint32_t lo = row_begin > face_row * FACE_ROWS ? row_begin : face_row * FACE_ROWS;
        const uint32_t hi = row_end < (face_row + 1) * FACE_ROWS ? row_end : (face_row + 1) * FACE_ROWS;
        if (lo >= hi) {
            continue;
        }
        for (uint32_t half = 0; half < 2; ++half) {
            if (half_mask & (1u << half)) {
                const uint32_t offset = tile_row_offset<face_row_bytes>(lo, half);
                list.append(offset, offset, (hi - lo) * face_row_bytes);
            }
        }
    }
    return list;
}

// Output rows [0, rows) of a page from a staging page whose row s holds output row (s - row_offset) mod 32.
// Pieces split wherever the source or the destination crosses a face row or the 32-row wrap: at most four
// pieces, eight runs.
template <uint32_t face_row_bytes>
RunList<8> rotated_runs(uint32_t row_offset, uint32_t rows) {
    RunList<8> list;
    uint32_t dst_row = 0;
    while (dst_row < rows) {
        const uint32_t src_row = (dst_row + row_offset) % TILE_ROWS;
        uint32_t len = rows - dst_row;
        const uint32_t dst_room = FACE_ROWS - dst_row % FACE_ROWS;
        const uint32_t src_room = FACE_ROWS - src_row % FACE_ROWS;
        len = len < dst_room ? len : dst_room;
        len = len < src_room ? len : src_room;
        for (uint32_t half = 0; half < 2; ++half) {
            list.append(
                tile_row_offset<face_row_bytes>(src_row, half),
                tile_row_offset<face_row_bytes>(dst_row, half),
                len * face_row_bytes);
        }
        dst_row += len;
    }
    return list;
}

}  // namespace g6_slice
