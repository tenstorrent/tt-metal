// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sampling geometry for fused multi-scale deformable attention, on the SFPU.
//
// Per point p and a block of up to 32 queries:
//
//   px   = (gx + 1) * x_scale - x_shift     x_scale/x_shift carry the
//   x0   = floor(px)                        align_corners variant, so this
//   dx   = px - x0                          kernel has no branch on it
//   w_c  = corner(dx, dy) * attn            for the four corners
//
// Queries occupy tile rows and only column 0 carries meaning, which is what
// mul_tiles_bcast<COL> consumes in the reduction — its scalar tiles were
// already 32 useful values out of 1024 before any of this.
//
// x0/y0 leave as bf16 and the reader decodes them with integer shifts, doing
// the corner indexing and bounds check itself. That is why nothing here needs
// int32 tiles, a typecast, a sentinel for out-of-bounds corners, or a 32-bit
// DST. It costs an exactness constraint: bf16 represents integers up to 256,
// so the caller must hold h_in and w_in at or below that.

#pragma once

#include <cstdint>

#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace msda_geometry {

// fp32 1.0 as the bit pattern the *_unary_tile scalars are passed as.
constexpr uint32_t ONE_BITS = 0x3F800000u;

// DST slots. Three are live at once in either window shape.
constexpr uint32_t DST_A = 0;
constexpr uint32_t DST_B = 1;
constexpr uint32_t DST_C = 2;

// One axis of one point: grid -> (floor, frac), both packed.
//
// floor and the subtraction share the window that produced px, so px never
// needs a scratch CB of its own.
inline void axis_split(
    uint32_t grid_cb,
    uint32_t floor_cb,
    uint32_t frac_cb,
    uint32_t scale_bits,
    uint32_t shift_bits,
    uint32_t& last_srca) {
    CircularBuffer grid(grid_cb);
    CircularBuffer floor_out(floor_cb);
    CircularBuffer frac_out(frac_cb);

    grid.wait_front(1);
    floor_out.reserve_back(1);
    frac_out.reserve_back(1);

    tile_regs_acquire();

    copy_tile_to_dst_init_short_with_dt(last_srca, grid_cb);
    last_srca = grid_cb;
    copy_tile(grid_cb, 0, DST_A);

    binop_with_scalar_tile_init();
    add_unary_tile(DST_A, ONE_BITS);
    mul_unary_tile(DST_A, scale_bits);
    sub_unary_tile(DST_A, shift_bits);  // DST_A = px

    copy_dest_values_init();
    copy_dest_values(DST_A, DST_B);
    rounding_op_tile_init();
    floor_tile(DST_B);  // DST_B = floor(px)

    sub_binary_tile_init();
    sub_binary_tile(DST_A, DST_B, DST_C);  // DST_C = frac

    tile_regs_commit();

    tile_regs_wait();
    pack_tile(DST_B, floor_cb);
    pack_tile(DST_C, frac_cb);
    tile_regs_release();

    grid.pop_front(1);
    floor_out.push_back(1);
    frac_out.push_back(1);
}

// One corner weight, folded with attn so the reduction's scalar tile arrives
// ready to broadcast. `invert_x`/`invert_y` pick which of dx / 1-dx and
// dy / 1-dy this corner uses.
//
// dx, dy and attn stay at the CB front across all four corners; the caller
// owns their wait_front / pop_front.
inline void corner_weight(
    uint32_t frac_x_cb,
    uint32_t frac_y_cb,
    uint32_t attn_cb,
    uint32_t scalar_cb,
    bool invert_x,
    bool invert_y,
    uint32_t& last_srca) {
    CircularBuffer scalar_out(scalar_cb);
    scalar_out.reserve_back(1);

    tile_regs_acquire();

    copy_tile_to_dst_init_short_with_dt(last_srca, frac_x_cb);
    last_srca = frac_x_cb;
    copy_tile(frac_x_cb, 0, DST_A);
    if (invert_x) {
        binop_with_scalar_tile_init();
        rsub_unary_tile(DST_A, ONE_BITS);
    }

    copy_tile_to_dst_init_short_with_dt(last_srca, frac_y_cb);
    last_srca = frac_y_cb;
    copy_tile(frac_y_cb, 0, DST_B);
    if (invert_y) {
        binop_with_scalar_tile_init();
        rsub_unary_tile(DST_B, ONE_BITS);
    }

    mul_binary_tile_init();
    mul_binary_tile(DST_A, DST_B, DST_C);

    // DST_A is free once the corner product is in DST_C.
    copy_tile_to_dst_init_short_with_dt(last_srca, attn_cb);
    last_srca = attn_cb;
    copy_tile(attn_cb, 0, DST_A);

    mul_binary_tile_init();
    mul_binary_tile(DST_C, DST_A, DST_C);

    tile_regs_commit();

    tile_regs_wait();
    pack_tile(DST_C, scalar_cb);
    tile_regs_release();

    scalar_out.push_back(1);
}

// Geometry for one point: two axis windows, then the four corners in the order
// the reduction consumes them (NW, NE, SW, SE).
inline void point(
    uint32_t grid_x_cb,
    uint32_t grid_y_cb,
    uint32_t attn_cb,
    uint32_t x0_cb,
    uint32_t y0_cb,
    uint32_t frac_x_cb,
    uint32_t frac_y_cb,
    uint32_t scalar_cb,
    uint32_t x_scale_bits,
    uint32_t x_shift_bits,
    uint32_t y_scale_bits,
    uint32_t y_shift_bits,
    uint32_t& last_srca) {
    axis_split(grid_x_cb, x0_cb, frac_x_cb, x_scale_bits, x_shift_bits, last_srca);
    axis_split(grid_y_cb, y0_cb, frac_y_cb, y_scale_bits, y_shift_bits, last_srca);

    CircularBuffer frac_x(frac_x_cb);
    CircularBuffer frac_y(frac_y_cb);
    CircularBuffer attn(attn_cb);

    frac_x.wait_front(1);
    frac_y.wait_front(1);
    attn.wait_front(1);

    corner_weight(frac_x_cb, frac_y_cb, attn_cb, scalar_cb, /*invert_x=*/true, /*invert_y=*/true, last_srca);
    corner_weight(frac_x_cb, frac_y_cb, attn_cb, scalar_cb, /*invert_x=*/false, /*invert_y=*/true, last_srca);
    corner_weight(frac_x_cb, frac_y_cb, attn_cb, scalar_cb, /*invert_x=*/true, /*invert_y=*/false, last_srca);
    corner_weight(frac_x_cb, frac_y_cb, attn_cb, scalar_cb, /*invert_x=*/false, /*invert_y=*/false, last_srca);

    frac_x.pop_front(1);
    frac_y.pop_front(1);
    attn.pop_front(1);
}

}  // namespace msda_geometry
