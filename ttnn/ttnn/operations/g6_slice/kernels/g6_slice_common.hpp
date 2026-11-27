// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Tile geometry and the face-row run lists shared by the g6_slice reader and writer.
//
// A 32x32 tile is four 16x16 faces stored face-major (0: rows 0-15 cols 0-15, 1: rows 0-15
// cols 16-31, 2: rows 16-31 cols 0-15, 3: rows 16-31 cols 16-31), each face row-major. A run of
// consecutive tile rows in one column half is therefore contiguous only while it stays inside one
// face; the builders below split there and merge whatever ends up contiguous again (a whole half
// tile is one run, a whole tile is one run).
//
// Run lists depend only on (row offset, rows in the tile), so the kernels build them once per row
// class at kernel start, never per page.

#include <stdint.h>

namespace g6_slice {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t TILE_COLS = 32;
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t LEFT_HALF = 1u << 0;
constexpr uint32_t RIGHT_HALF = 1u << 1;
constexpr uint32_t BOTH_HALVES = LEFT_HALF | RIGHT_HALF;

// One 16-row segment x two column halves.
constexpr uint32_t MAX_BAND_RUNS = 4;
// [0, rv) splits where the output row crosses 16 or the staging row crosses 16 or wraps at 32:
// at most four pieces, two column halves each.
constexpr uint32_t MAX_STORE_RUNS = 8;

template <uint32_t page_bytes>
struct TileGeometry {
    static constexpr uint32_t face_bytes = page_bytes / 4;
    static constexpr uint32_t face_row_bytes = face_bytes / FACE_ROWS;

    static constexpr uint32_t row_offset(uint32_t row, uint32_t half) {
        return ((row / FACE_ROWS) * 2 + half) * face_bytes + (row % FACE_ROWS) * face_row_bytes;
    }
};

// Byte runs of tile rows [row_begin, row_end) in the column halves of `half_mask`, at their natural
// in-tile offsets (the DRAM read source and the L1 staging destination share the offset).
struct BandRuns {
    uint32_t count = 0;
    uint32_t offset[MAX_BAND_RUNS];
    uint32_t bytes[MAX_BAND_RUNS];

    void append(uint32_t run_offset, uint32_t run_bytes) {
        if (count > 0 && offset[count - 1] + bytes[count - 1] == run_offset) {
            bytes[count - 1] += run_bytes;
            return;
        }
        offset[count] = run_offset;
        bytes[count] = run_bytes;
        ++count;
    }
};

template <uint32_t page_bytes>
BandRuns make_band_runs(uint32_t row_begin, uint32_t row_end, uint32_t half_mask) {
    using Geometry = TileGeometry<page_bytes>;
    BandRuns runs;
    for (uint32_t segment_begin = row_begin; segment_begin < row_end;) {
        const uint32_t face_end = (segment_begin / FACE_ROWS + 1) * FACE_ROWS;
        const uint32_t segment_end = row_end < face_end ? row_end : face_end;
        const uint32_t segment_bytes = (segment_end - segment_begin) * Geometry::face_row_bytes;
        for (uint32_t half = 0; half < 2; ++half) {
            if (half_mask & (1u << half)) {
                runs.append(Geometry::row_offset(segment_begin, half), segment_bytes);
            }
        }
        segment_begin = segment_end;
    }
    return runs;
}

// Output rows [0, rows_in_tile) of a page, taken from staging rows (i + row_offset) mod 32.
struct StoreRuns {
    uint32_t count = 0;
    uint32_t src_offset[MAX_STORE_RUNS];
    uint32_t dst_offset[MAX_STORE_RUNS];
    uint32_t bytes[MAX_STORE_RUNS];

    void append(uint32_t src, uint32_t dst, uint32_t run_bytes) {
        if (count > 0 && src_offset[count - 1] + bytes[count - 1] == src &&
            dst_offset[count - 1] + bytes[count - 1] == dst) {
            bytes[count - 1] += run_bytes;
            return;
        }
        src_offset[count] = src;
        dst_offset[count] = dst;
        bytes[count] = run_bytes;
        ++count;
    }
};

template <uint32_t page_bytes>
StoreRuns make_store_runs(uint32_t row_offset, uint32_t rows_in_tile) {
    using Geometry = TileGeometry<page_bytes>;
    StoreRuns runs;
    for (uint32_t out_row = 0; out_row < rows_in_tile;) {
        const uint32_t staging_row = (out_row + row_offset) % TILE_ROWS;
        uint32_t rows = rows_in_tile - out_row;
        const uint32_t out_face_left = FACE_ROWS - out_row % FACE_ROWS;
        const uint32_t staging_face_left = FACE_ROWS - staging_row % FACE_ROWS;
        rows = rows < out_face_left ? rows : out_face_left;
        rows = rows < staging_face_left ? rows : staging_face_left;
        for (uint32_t half = 0; half < 2; ++half) {
            runs.append(
                Geometry::row_offset(staging_row, half),
                Geometry::row_offset(out_row, half),
                rows * Geometry::face_row_bytes);
        }
        out_row += rows;
    }
    return runs;
}

}  // namespace g6_slice
