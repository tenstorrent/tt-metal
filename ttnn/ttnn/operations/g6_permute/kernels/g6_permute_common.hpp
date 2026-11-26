// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the rows_tile reader and writer: compile-time plan, block decode, tile-row geometry.
// Both kernels receive the same CT-arg prefix (see g6_permute_program_descriptor.py, _plan_ct_args),
// so a block index decodes to the same (n, a-group, b-group, wt-group) on both RISC-Vs.

#pragma once

#include <cstdint>

namespace g6_permute {

// Index space of x [N, A, B, W] -> out [N, B, A, W]; counts per image.
constexpr uint32_t A = get_compile_time_arg_val(0);
constexpr uint32_t B = get_compile_time_arg_val(1);
constexpr uint32_t At = get_compile_time_arg_val(2);
constexpr uint32_t Bt = get_compile_time_arg_val(3);
constexpr uint32_t Wt = get_compile_time_arg_val(4);

// Block knobs (host select_plan / BLOCK_WT) and the block grid derived from them.
constexpr uint32_t block_a = get_compile_time_arg_val(5);
constexpr uint32_t block_b = get_compile_time_arg_val(6);
constexpr uint32_t block_wt = get_compile_time_arg_val(7);
constexpr uint32_t num_a_groups = get_compile_time_arg_val(8);
constexpr uint32_t num_b_groups = get_compile_time_arg_val(9);
constexpr uint32_t num_wt_groups = get_compile_time_arg_val(10);

// Tile geometry in bytes and rows.
constexpr uint32_t tile_bytes = get_compile_time_arg_val(11);
constexpr uint32_t tile_height = get_compile_time_arg_val(12);
constexpr uint32_t face_height = get_compile_time_arg_val(13);
constexpr uint32_t face_cols = get_compile_time_arg_val(14);  // faces side by side in one tile row
constexpr uint32_t face_bytes = get_compile_time_arg_val(15);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(16);

// Buffer depths and CB slots.
constexpr uint32_t depth_in = get_compile_time_arg_val(17);
constexpr uint32_t depth_staging = get_compile_time_arg_val(18);
constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(19);
constexpr uint32_t cb_output_staging = get_compile_time_arg_val(20);

constexpr uint32_t num_plan_ct_args = 21;

constexpr bool regroup = block_a > 1;
constexpr uint32_t input_pages_per_block = block_a * block_wt;
constexpr uint32_t staging_pages_per_block = block_b * block_wt;

static_assert(block_a >= 1 && block_a <= tile_height && tile_height % block_a == 0);
static_assert(block_b >= 1 && block_b <= tile_height && tile_height % block_b == 0);
// A span below a whole tile must stay inside one face band, so its rows form one run per face column.
static_assert(block_a == tile_height || face_height % block_a == 0);
static_assert(block_b == tile_height || face_height % block_b == 0);
static_assert(depth_in >= 1);
static_assert(!regroup || depth_staging >= 1);

FORCE_INLINE constexpr uint32_t min_u32(uint32_t x, uint32_t y) { return x < y ? x : y; }

// Byte offset of face column `h` of tile row `r`: faces are row-major inside the tile, rows inside a face.
FORCE_INLINE constexpr uint32_t face_off(uint32_t r, uint32_t h) {
    return ((r / face_height) * face_cols + h) * face_bytes + (r % face_height) * face_row_bytes;
}

// Calls fn(offset, bytes) once per contiguous byte run that holds tile rows [r0, r0 + n_rows).
// A run never crosses a face band; a span crossing one is split at the band edge.
template <typename Fn>
FORCE_INLINE void for_each_row_run(uint32_t r0, uint32_t n_rows, Fn&& fn) {
    while (n_rows > 0) {
        const uint32_t band_end = (r0 / face_height + 1) * face_height;
        const uint32_t rows = min_u32(n_rows, band_end - r0);
        for (uint32_t h = 0; h < face_cols; ++h) {
            fn(face_off(r0, h), rows * face_row_bytes);
        }
        r0 += rows;
        n_rows -= rows;
    }
}

// One block = one image, block_a rows of a, block_b rows of b, block_wt tile columns.
// live_* are the ragged counts of the last group along each axis.
struct Block {
    uint32_t n;
    uint32_t a0, live_a;
    uint32_t b0, live_b;
    uint32_t wt0, live_wt;
};

FORCE_INLINE Block decode_block(uint32_t block_idx) {
    // block_idx = ((n * NA + ag) * NBg + bg) * NW + wtb
    const uint32_t wtb = block_idx % num_wt_groups;
    block_idx /= num_wt_groups;
    const uint32_t bg = block_idx % num_b_groups;
    block_idx /= num_b_groups;
    const uint32_t ag = block_idx % num_a_groups;
    const uint32_t n = block_idx / num_a_groups;

    Block blk;
    blk.n = n;
    blk.a0 = ag * block_a;
    blk.live_a = min_u32(block_a, A - blk.a0);
    blk.b0 = bg * block_b;
    blk.live_b = min_u32(block_b, B - blk.b0);
    blk.wt0 = wtb * block_wt;
    blk.live_wt = min_u32(block_wt, Wt - blk.wt0);
    return blk;
}

FORCE_INLINE uint32_t in_page(uint32_t n, uint32_t a, uint32_t bt, uint32_t wt) {
    return ((n * A + a) * Bt + bt) * Wt + wt;
}

FORCE_INLINE uint32_t out_page(uint32_t n, uint32_t b, uint32_t at, uint32_t wt) {
    return ((n * B + b) * At + at) * Wt + wt;
}

// Live rows of tile row `t` along an axis of `extent` rows.
FORCE_INLINE constexpr uint32_t live_rows_of_tile_row(uint32_t extent, uint32_t t) {
    return min_u32(tile_height, extent - t * tile_height);
}

}  // namespace g6_permute
