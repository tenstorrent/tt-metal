// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "g6_split_mode.hpp"  // [small_tensor_split] constexpr G6_SPLIT_MODE

// Geometry, block knobs and the block walk shared by every reader and writer of g6_to_layout.
// The walk must be identical in the reader and the writer of a regime: both cut the core's
// tile range into the same blocks, and the CB carries exactly one page per block.

namespace g6_to_layout {

constexpr uint32_t TILE_HW = 32;
constexpr uint32_t FACE_HW = 16;
constexpr uint32_t DATUM_BYTES = 2;  // bfloat16
constexpr uint32_t FACE_ROW_BYTES = FACE_HW * DATUM_BYTES;
constexpr uint32_t FACE_BYTES = FACE_HW * FACE_ROW_BYTES;
constexpr uint32_t TILE_BYTES = 4 * FACE_BYTES;
// R2 slot of one tile: faces 0 and 2 only (faces 1 and 3 hold no live column when W <= 16).
constexpr uint32_t LEFT_SLOT_BYTES = 2 * FACE_BYTES;
// Bytes one tile column contributes to an output row.
constexpr uint32_t TILE_ROW_BYTES = TILE_HW * DATUM_BYTES;

// Compile-time args 0..6 of every kernel; kernel-specific args start at `next_ct_arg`.
struct Geometry {
    static constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(0);
    static constexpr uint32_t image_rows_tiles = get_compile_time_arg_val(1);
    static constexpr uint32_t height = get_compile_time_arg_val(2);
    static constexpr uint32_t width = get_compile_time_arg_val(3);
    static constexpr uint32_t block_rows_tiles = get_compile_time_arg_val(4);
    static constexpr uint32_t block_cols_tiles = get_compile_time_arg_val(5);
    // Rows a tile-row occupies in the row stage: min(32, H).
    static constexpr uint32_t tile_row_stage_rows = get_compile_time_arg_val(6);
    static constexpr uint32_t next_ct_arg = 7;

    static constexpr bool full_row_blocks = block_cols_tiles == tensor_cols_tiles;
    static constexpr uint32_t stage_row_bytes = block_cols_tiles * TILE_ROW_BYTES;
    static constexpr uint32_t stage_tile_row_bytes = tile_row_stage_rows * stage_row_bytes;
};

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Rows of tensor tile-row `g` that hold logical values (each image is tile-padded on its own).
FORCE_INLINE uint32_t live_rows_of(uint32_t g) {
    const uint32_t ht = g % Geometry::image_rows_tiles;
    return min_u32(TILE_HW, Geometry::height - ht * TILE_HW);
}

// ROW_MAJOR page of row 0 of tensor tile-row `g` (the output's in TILE -> RM, the input's in
// RM -> TILE): pages are contiguous across images.
FORCE_INLINE uint32_t first_row_page_of(uint32_t g) {
    const uint32_t image = g / Geometry::image_rows_tiles;
    const uint32_t ht = g - image * Geometry::image_rows_tiles;
    return image * Geometry::height + ht * TILE_HW;
}

// Columns of tile column `c` that hold logical values.
FORCE_INLINE uint32_t live_cols_of(uint32_t c) { return min_u32(TILE_HW, Geometry::width - c * TILE_HW); }

// A rectangle of tiles: `rows` tile-rows from `tile_row`, `cols` tile columns from `col`.
// Ragged extents are runtime values; the CB page stays the nominal block.
struct Block {
    uint32_t tile_row;
    uint32_t rows;
    uint32_t col;
    uint32_t cols;
    // [small_tensor_split] Row band inside every tile-row of the block: rows [row_lo, row_hi),
    // clamped to the tile-row's live rows where used. (0, 32) = the whole tile-row (the op's
    // behaviour); every pre-existing aggregate init keeps it.
    uint32_t row_lo = 0;
    uint32_t row_hi = TILE_HW;
};

// [small_tensor_split] Last row (exclusive) of the block's band in tile-row `g`.
FORCE_INLINE uint32_t band_hi_of(const Block& block, uint32_t g) { return min_u32(block.row_hi, live_rows_of(g)); }

// Live bytes of one ROW_MAJOR row inside the block's columns; they start at byte col * 64 of the
// page and end at the row's last live byte, so padding columns never cross the NoC.
FORCE_INLINE uint32_t row_segment_bytes(const Block& block) {
    const uint32_t segment_end = min_u32(Geometry::width, (block.col + block.cols) * TILE_HW) * DATUM_BYTES;
    return segment_end - block.col * TILE_ROW_BYTES;
}

// Cuts the core's flattened tile range [begin, end) into blocks. With full-row blocks a block
// is a run of up to block_rows_tiles whole tile-rows and a partial head/tail row is its own
// block; otherwise every row span is cut into blocks of up to block_cols_tiles columns.
struct BlockWalk {
    uint32_t tile;
    uint32_t end;

    FORCE_INLINE bool next(Block& block) {
        using G = Geometry;
        if (tile >= end) {
            return false;
        }
        const uint32_t g = tile / G::tensor_cols_tiles;
        const uint32_t col = tile - g * G::tensor_cols_tiles;
        const uint32_t span_cols = min_u32(G::tensor_cols_tiles - col, end - tile);
        if (G::full_row_blocks && col == 0 && span_cols == G::tensor_cols_tiles) {
            const uint32_t rows = min_u32(G::block_rows_tiles, (end - tile) / G::tensor_cols_tiles);
            block = Block{g, rows, 0, G::tensor_cols_tiles};
            tile += rows * G::tensor_cols_tiles;
        } else {
            const uint32_t cols = min_u32(G::block_cols_tiles, span_cols);
            block = Block{g, 1, col, cols};
            tile += cols;
        }
        return true;
    }
};

// Cuts one resident shard's live rectangle into blocks of up to block_rows_tiles tile-rows, each
// spanning the shard's live columns. Only the last block may be short.
struct RectWalk {
    uint32_t tile_row;
    uint32_t rows_left;
    uint32_t col;
    uint32_t cols;

    FORCE_INLINE bool next(Block& block) {
        if (rows_left == 0) {
            return false;
        }
        const uint32_t rows = min_u32(Geometry::block_rows_tiles, rows_left);
        block = Block{tile_row, rows, col, cols};
        tile_row += rows;
        rows_left -= rows;
        return true;
    }
};

// The blocks of this core. Interleaved runtime args: (tile_begin, num_tiles) of the flattened
// tile split. Sharded runtime args: (tile_row, rows, col, cols) of the core's live shard rectangle.
template <bool sharded>
struct CoreBlocks;

template <>
struct CoreBlocks<false> {
    BlockWalk walk{get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1)};
    FORCE_INLINE bool next(Block& block) { return walk.next(block); }
};

template <>
struct CoreBlocks<true> {
    RectWalk walk{
        get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};
    FORCE_INLINE bool next(Block& block) { return walk.next(block); }
};

// Tile-rows a block moves through a tile-paged CB. Interleaved blocks always move the nominal
// block_rows_tiles so every block occupies the same ring span; a shard moves only its live rows
// (its last block is the core's last, so the ring never needs to stay aligned after it).
template <bool sharded>
FORCE_INLINE uint32_t moved_tile_rows(const Block& block) {
    return sharded ? block.rows : Geometry::block_rows_tiles;
}

// ---------------------------------------------------------------------------------------------
// [small_tensor_split] The interleaved walk of the experiment kernels, chosen at compile time by
// G6_SPLIT_MODE (g6_split_mode.hpp, one kernel directory per mode):
//   mode 0: runtime args (tile_begin, num_tiles): the op's CoreBlocks<false>, unchanged.
//   mode 1: runtime args (tile_begin, num_tiles, row_lo, row_hi, sub_rows): the same tile range,
//           every tile-row cut to the row band [row_lo, row_hi); a one-tile-row block is further
//           cut into blocks of at most sub_rows rows (same tiles, successive sub-bands), so the
//           writer drains sub-band k while the reader reorders sub-band k + 1 of the same tiles.
//   mode 3: mode 1 without sub-bands (lean band walk), see BandCoreBlocksT<3>.
//   mode 2: runtime args (unit_begin, unit_end, band_rows): a flattened range of OUTPUT row
//           segments (unit = R * tensor_cols_tiles + c, R = output page); needs
//           block_cols_tiles == tensor_cols_tiles.
// Reader and writer construct the same walk from the same args, so their blocks are identical.
struct RowUnitWalk {
    uint32_t u;
    uint32_t end;
    uint32_t band_rows;

    FORCE_INLINE bool next(Block& block) {
        using G = Geometry;
        if (u >= end) {
            return false;
        }
        const uint32_t page = u / G::tensor_cols_tiles;
        const uint32_t c = u - page * G::tensor_cols_tiles;
        const uint32_t image = page / G::height;
        const uint32_t h = page - image * G::height;
        const uint32_t ht = h / TILE_HW;
        const uint32_t g = image * G::image_rows_tiles + ht;
        const uint32_t r = h - ht * TILE_HW;
        const uint32_t left = end - u;
        if (c != 0 || left < G::tensor_cols_tiles) {
            const uint32_t cols = min_u32(min_u32(G::tensor_cols_tiles - c, left), G::block_cols_tiles);
            block = Block{g, 1, c, cols, r, r + 1};
            u += cols;
        } else {
            const uint32_t live = live_rows_of(g);
            const uint32_t rows = min_u32(min_u32(live - r, left / G::tensor_cols_tiles), band_rows);
            block = Block{g, 1, 0, G::tensor_cols_tiles, r, r + rows};
            u += rows * G::tensor_cols_tiles;
        }
        return true;
    }
};

struct SubBandWalk {
    BlockWalk walk;
    uint32_t row_lo;
    uint32_t row_hi;
    uint32_t sub_rows;
    Block cur{};
    uint32_t next_lo = 0;
    uint32_t cur_hi = 0;

    FORCE_INLINE bool next(Block& block) {
        if (next_lo >= cur_hi) {
            if (!walk.next(cur)) {
                return false;
            }
            next_lo = row_lo;
            // A one-tile-row block is cut into sub-bands up to its live rows; a taller block keeps
            // the band whole.
            cur_hi = cur.rows == 1 ? min_u32(row_hi, live_rows_of(cur.tile_row)) : row_hi;
            if (next_lo >= cur_hi) {  // empty band in this tile-row: one empty block keeps both walks equal
                block = cur;
                block.row_lo = row_lo;
                block.row_hi = row_lo;
                return true;
            }
        }
        block = cur;
        block.row_lo = next_lo;
        block.row_hi = cur.rows == 1 ? min_u32(next_lo + sub_rows, cur_hi) : cur_hi;
        next_lo = block.row_hi;
        return true;
    }
};

template <uint32_t mode>
struct BandCoreBlocksT;

template <>
struct BandCoreBlocksT<0> {
    BlockWalk walk{get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1)};
    FORCE_INLINE bool next(Block& block) { return walk.next(block); }
};

template <>
struct BandCoreBlocksT<1> {
    SubBandWalk walk{
        {get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1)},
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4)};
    FORCE_INLINE bool next(Block& block) { return walk.next(block); }
};

template <>
struct BandCoreBlocksT<2> {
    RowUnitWalk walk{get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)};
    FORCE_INLINE bool next(Block& block) { return walk.next(block); }
};

// mode 3 (lean band): runtime args (tile_begin, num_tiles, row_lo, row_hi): mode 1 without
// sub-bands - the op's BlockWalk, every block carrying the core's one row band.
template <>
struct BandCoreBlocksT<3> {
    BlockWalk walk{get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(1)};
    uint32_t row_lo = get_arg_val<uint32_t>(2);
    uint32_t row_hi = get_arg_val<uint32_t>(3);
    FORCE_INLINE bool next(Block& block) {
        if (!walk.next(block)) {
            return false;
        }
        block.row_lo = row_lo;
        block.row_hi = row_hi;
        return true;
    }
};

using BandCoreBlocks = BandCoreBlocksT<G6_SPLIT_MODE>;

}  // namespace g6_to_layout
