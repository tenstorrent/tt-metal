// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the rows_tile reader and writer: the compile-time-arg prefix both kernels receive in the
// same order, the block grid of op_design.md, and the tile-row byte geometry.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_permute {

// Logical extents of x = [N, A, B, W]; WT = ceil(W / tile width).
constexpr uint32_t A = get_compile_time_arg_val(0);
constexpr uint32_t B = get_compile_time_arg_val(1);
constexpr uint32_t WT = get_compile_time_arg_val(2);
// Page strides come from the padded shapes: interleaved pages are laid over them, not over the logical shape.
constexpr uint32_t IN_A = get_compile_time_arg_val(3);
constexpr uint32_t IN_BT = get_compile_time_arg_val(4);
constexpr uint32_t IN_WT = get_compile_time_arg_val(5);
constexpr uint32_t OUT_B = get_compile_time_arg_val(6);
constexpr uint32_t OUT_AT = get_compile_time_arg_val(7);
constexpr uint32_t OUT_WT = get_compile_time_arg_val(8);
constexpr uint32_t BLOCK_A = get_compile_time_arg_val(9);
constexpr uint32_t BLOCK_B = get_compile_time_arg_val(10);
constexpr uint32_t BLOCK_WT = get_compile_time_arg_val(11);
constexpr uint32_t TILE_H = get_compile_time_arg_val(12);
constexpr uint32_t FACE_H = get_compile_time_arg_val(13);
constexpr uint32_t FACES_PER_BAND = get_compile_time_arg_val(14);
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(15);
constexpr uint32_t FACE_BYTES = get_compile_time_arg_val(16);
constexpr uint32_t FACE_ROW_BYTES = get_compile_time_arg_val(17);
constexpr uint32_t CB_INPUT_TILES = get_compile_time_arg_val(18);
constexpr uint32_t NUM_SHARED_CT_ARGS = 19;

constexpr uint32_t NA = (A + BLOCK_A - 1) / BLOCK_A;
constexpr uint32_t NBG = (B + BLOCK_B - 1) / BLOCK_B;
constexpr uint32_t NW = (WT + BLOCK_WT - 1) / BLOCK_WT;
// Ragged blocks still push and pop the nominal count so the input ring closes on every core.
constexpr uint32_t INPUT_PAGES_PER_BLOCK = BLOCK_A * BLOCK_WT;

// A row span of a sub-tile block must stay inside one face band: then each face half of the span
// is one contiguous run. Blocks of TILE_H rows always cover the tile's live rows and move whole.
static_assert(TILE_H % BLOCK_A == 0 && TILE_H % BLOCK_B == 0, "block_a and block_b must divide the tile height");
static_assert(BLOCK_A == TILE_H || FACE_H % BLOCK_A == 0, "a sub-tile a-span must not cross the face band");
static_assert(BLOCK_B == TILE_H || FACE_H % BLOCK_B == 0, "a sub-tile b-span must not cross the face band");

struct Block {
    uint32_t n;
    uint32_t a0;
    uint32_t live_a;
    uint32_t b0;
    uint32_t live_b;
    uint32_t wt0;
    uint32_t live_wt;
};

FORCE_INLINE uint32_t min_u32(uint32_t x, uint32_t y) { return x < y ? x : y; }

// block_idx = ((n * NA + ag) * NBG + bg) * NW + wtb
FORCE_INLINE Block decode_block(uint32_t block_idx) {
    const uint32_t wtb = block_idx % NW;
    block_idx /= NW;
    const uint32_t bg = block_idx % NBG;
    block_idx /= NBG;
    const uint32_t ag = block_idx % NA;
    const uint32_t n = block_idx / NA;
    Block blk;
    blk.n = n;
    blk.a0 = ag * BLOCK_A;
    blk.live_a = min_u32(BLOCK_A, A - blk.a0);
    blk.b0 = bg * BLOCK_B;
    blk.live_b = min_u32(BLOCK_B, B - blk.b0);
    blk.wt0 = wtb * BLOCK_WT;
    blk.live_wt = min_u32(BLOCK_WT, WT - blk.wt0);
    return blk;
}

FORCE_INLINE uint32_t in_page(uint32_t n, uint32_t a, uint32_t bt, uint32_t wt) {
    return ((n * IN_A + a) * IN_BT + bt) * IN_WT + wt;
}

FORCE_INLINE uint32_t out_page(uint32_t n, uint32_t b, uint32_t at, uint32_t wt) {
    return ((n * OUT_B + b) * OUT_AT + at) * OUT_WT + wt;
}

// Byte offset of half `h` of tile row `r`: a tile row is FACES_PER_BAND face rows, one per face of its band.
FORCE_INLINE uint32_t face_off(uint32_t r, uint32_t h) {
    return ((r / FACE_H) * FACES_PER_BAND + h) * FACE_BYTES + (r % FACE_H) * FACE_ROW_BYTES;
}

// True when rows [r0, r0 + rows) hold every live row of a tile whose dim has `remaining` rows from
// the tile's first row on; the tile then moves as one page.
FORCE_INLINE bool covers_live_rows(uint32_t r0, uint32_t rows, uint32_t remaining) {
    return r0 == 0 && rows >= min_u32(TILE_H, remaining);
}

// Calls fn(offset, bytes) once per face half for rows [r0, r0 + rows) of a tile; the span lies in one
// face band (static_asserts above), so each half is one contiguous run at the same tile-relative offset
// in source and destination.
template <typename Fn>
FORCE_INLINE void for_each_row_run(uint32_t r0, uint32_t rows, Fn&& fn) {
    for (uint32_t h = 0; h < FACES_PER_BAND; ++h) {
        fn(face_off(r0, h), rows * FACE_ROW_BYTES);
    }
}

}  // namespace g6_permute
