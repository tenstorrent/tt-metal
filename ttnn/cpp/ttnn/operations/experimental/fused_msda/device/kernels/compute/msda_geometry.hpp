// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sampling geometry for fused multi-scale deformable attention, on the SFPU.
//
// Why the geometry runs here
// --------------------------
// It is per-point float work over 32 query rows: vector work. The dataflow RISC
// has no FPU, so evaluating it there costs ~140 cycles per operation in
// soft-float emulation. The SFPU evaluates the same expression across the tile.
//
// What one point costs
// --------------------
// Per sampling point (one (level, point) pair) and a block of up to 32 queries:
//
//   px   = primary * primary_scale + secondary * secondary_scale + bias
//   x0   = floor(px)
//   dx   = px - x0
//   w_c  = corner(dx, dy) * attn                        for the four corners
//
// `primary` is the normalized x (V1) or the reference point's x (V2);
// `secondary` is the raw sampling offset and exists only for V2. The
// (align_corners, locations_in_grid_space, from_offsets) variants are folded
// into the three per-level constants on the host, so this kernel is branch-free
// on them — see fused_msda_program_factory.cpp::axis_constants.
//
// Queries occupy tile rows and only column 0 carries meaning, which is what
// `mul_tiles_bcast<COL>` consumes in the reduction: a scalar tile is 32 useful
// values out of 1024.
//
// x0/y0 leave as bf16 and the reader decodes them with integer shifts, doing the
// corner indexing and the bounds test itself. Nothing here needs int32 tiles, a
// typecast, or a sentinel for an out-of-bounds corner. The constraint that buys
// is exactness: bf16 holds every integer up to 256, so each level's H and W must
// be at or below that. `derive_shapes` in fused_msda_device_operation.cpp
// rejects anything larger.
//
// fp32_dest_acc_en is required, not an optimization. px reaches the feature
// map's extent, and bf16's ulp at 200 is 1.0 — in a 16-bit destination
// floor(px) on the 200x113 level rounds to the wrong integer and the fraction it
// feeds collapses, degrading bilinear sampling to nearest-neighbour.

#pragma once

#include <cstdint>

#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace fused_msda_geometry {

// fp32 1.0 as the bit pattern the *_unary_tile scalars are passed as.
constexpr uint32_t ONE_BITS = 0x3F800000u;

// DST slots. `axis()` on the from-offsets path writes all four inside one
// acquire/release window, which is the entire budget a 32-bit destination
// allows under DstSync::Half (tech_reports/matrix_engine/matrix_engine.md:83;
// `dest_size = is_fp32_dest_acc_en ? 4 : 8` in api/compute/tilize.h). There is
// no spare slot: another temporary means restructuring the window, not taking
// index 4.
constexpr uint32_t DST_PX = 0;
constexpr uint32_t DST_FLOOR = 1;
constexpr uint32_t DST_FRAC = 2;
constexpr uint32_t DST_AUX = 3;
constexpr uint32_t DST_FP32_CAPACITY = 4;
static_assert(DST_AUX < DST_FP32_CAPACITY, "geometry uses a DST slot a 32-bit destination does not have");

// The host-folded normalized -> pixel mapping for one axis of one level:
//   px = primary * primary_scale + secondary * secondary_scale + bias
// `secondary_scale` is unused when the op runs without sampling offsets. All
// three are fp32 bit patterns; must stay in step with `AxisConstants` in
// fused_msda_program_factory.cpp, which builds them.
struct AxisConstants {
    uint32_t primary_scale;
    uint32_t secondary_scale;
    uint32_t bias;
};

// The circular buffers one geometry pass moves through. Grouped rather than
// passed positionally: ten same-typed CB indices in a row let a transposition
// (x0 for y0, frac_x for frac_y) compile cleanly and produce transposed
// sampling that PCC survives on symmetric shapes.
struct GeometryPipes {
    uint32_t geom_x;     // reader -> compute, the primary operand
    uint32_t geom_y;
    uint32_t offset_x;   // reader -> compute, the secondary operand (from-offsets only)
    uint32_t offset_y;
    uint32_t attn;       // reader -> compute
    uint32_t x0;         // compute -> reader, floor(px)
    uint32_t y0;
    uint32_t frac_x;     // compute -> compute
    uint32_t frac_y;
    uint32_t scalar;     // compute -> compute, consumed by the reduction
};

// Makes SrcA carry `new_cb` for a datacopy. `srca_cb` tracks what SrcA was last
// configured for so the reconfiguration is skipped when nothing changed; it must
// be seeded with whatever `compute_kernel_hw_startup` left SrcA on.
inline void copy_from(uint32_t new_cb, uint32_t& srca_cb) {
    reconfig_data_format_srca(srca_cb, new_cb);
    copy_init(new_cb);
    srca_cb = new_cb;
}

// One axis of one point: (primary, secondary) -> (floor(px), px - floor(px)),
// both packed for downstream consumers.
//
// Both results come out of a single DST window. They land in two different
// circular buffers, which is legal because every CB this kernel packs to holds
// bf16 32x32 tiles — the packer is configured per format, not per destination.
template <bool FROM_OFFSETS>
inline void axis(
    uint32_t primary_cb,
    uint32_t secondary_cb,
    uint32_t floor_cb,
    uint32_t frac_cb,
    const AxisConstants& k,
    uint32_t& srca_cb) {
    CircularBuffer floor_out(floor_cb);
    CircularBuffer frac_out(frac_cb);
    floor_out.reserve_back(1);
    frac_out.reserve_back(1);

    tile_regs_acquire();

    // Every copy_tile first, then the maths. Interleaving an SFPU op between two
    // copies reconfigures the unpacker mid-window.
    copy_from(primary_cb, srca_cb);
    copy_tile(primary_cb, 0, DST_PX);
    if constexpr (FROM_OFFSETS) {
        copy_from(secondary_cb, srca_cb);
        copy_tile(secondary_cb, 0, DST_AUX);
    }

    binop_with_scalar_tile_init();
    mul_unary_tile(DST_PX, k.primary_scale);
    if constexpr (FROM_OFFSETS) {
        mul_unary_tile(DST_AUX, k.secondary_scale);
        add_binary_tile_init();
        add_binary_tile(DST_PX, DST_AUX, DST_PX);
        binop_with_scalar_tile_init();
    }
    add_unary_tile(DST_PX, k.bias);  // DST_PX = px

    copy_dest_values_init();
    // Float32 because fp32_dest_acc_en makes DST 32-bit; px is exactly the
    // value that needs those bits (see the header note).
    copy_dest_values<DataFormat::Float32>(DST_PX, DST_FLOOR);
    rounding_op_tile_init();
    floor_tile(DST_FLOOR);  // DST_FLOOR = floor(px)

    sub_binary_tile_init();
    sub_binary_tile(DST_PX, DST_FLOOR, DST_FRAC);  // DST_FRAC = px - floor(px)

    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(DST_FLOOR, floor_cb, 0);
    pack_tile<true>(DST_FRAC, frac_cb, 0);
    tile_regs_release();

    floor_out.push_back(1);
    frac_out.push_back(1);
}

// One corner weight, folded with attn so the reduction's scalar tile arrives
// ready to broadcast. `invert_x` / `invert_y` pick which of dx / 1-dx and
// dy / 1-dy this corner uses.
//
// dx, dy and attn stay at their CB fronts across all four corners; the caller
// owns their wait_front / pop_front.
inline void corner_weight(
    uint32_t frac_x_cb,
    uint32_t frac_y_cb,
    uint32_t attn_cb,
    uint32_t scalar_cb,
    bool invert_x,
    bool invert_y,
    uint32_t& srca_cb) {
    // The DST slots hold fractions and a weight here, not the quantities their
    // names describe in axis(). Alias them so the multiplies read as what they
    // compute.
    constexpr uint32_t DST_DX = DST_PX;
    constexpr uint32_t DST_DY = DST_FLOOR;
    constexpr uint32_t DST_ATTN = DST_FRAC;

    CircularBuffer scalar_out(scalar_cb);
    scalar_out.reserve_back(1);

    tile_regs_acquire();

    copy_from(frac_x_cb, srca_cb);
    copy_tile(frac_x_cb, 0, DST_DX);
    copy_from(frac_y_cb, srca_cb);
    copy_tile(frac_y_cb, 0, DST_DY);
    copy_from(attn_cb, srca_cb);
    copy_tile(attn_cb, 0, DST_ATTN);

    if (invert_x || invert_y) {
        binop_with_scalar_tile_init();
        if (invert_x) {
            rsub_unary_tile(DST_DX, ONE_BITS);
        }
        if (invert_y) {
            rsub_unary_tile(DST_DY, ONE_BITS);
        }
    }

    mul_binary_tile_init();
    mul_binary_tile(DST_DX, DST_DY, DST_DX);
    mul_binary_tile(DST_DX, DST_ATTN, DST_DX);

    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(DST_DX, scalar_cb, 0);
    tile_regs_release();

    scalar_out.push_back(1);
}

// Geometry for one sampling point: the two axes, then the four corner scalars.
//
// The corner order is NW, NE, SW, SE, and it is a contract with the reader: the
// four scalar tiles pushed here are consumed against the four input-tile groups
// the reader gathers with `dy_off = (c < 2) ? 0 : 1; dx_off = (c & 1) ? 1 : 0`
// in fused_msda_reader_common.hpp. The two orders must agree element for
// element or every sample is silently wrong.
//
// The floors go to the reader, which turns them into page indices; the
// fractions never leave the compute kernel.
template <bool FROM_OFFSETS>
inline void point(
    const GeometryPipes& cb, const AxisConstants& kx, const AxisConstants& ky, uint32_t& srca_cb) {
    CircularBuffer geom_x(cb.geom_x);
    CircularBuffer geom_y(cb.geom_y);
    CircularBuffer offset_x(cb.offset_x);
    CircularBuffer offset_y(cb.offset_y);
    CircularBuffer attn(cb.attn);
    CircularBuffer frac_x(cb.frac_x);
    CircularBuffer frac_y(cb.frac_y);

    geom_x.wait_front(1);
    geom_y.wait_front(1);
    attn.wait_front(1);
    if constexpr (FROM_OFFSETS) {
        offset_x.wait_front(1);
        offset_y.wait_front(1);
    }

    axis<FROM_OFFSETS>(cb.geom_x, cb.offset_x, cb.x0, cb.frac_x, kx, srca_cb);
    axis<FROM_OFFSETS>(cb.geom_y, cb.offset_y, cb.y0, cb.frac_y, ky, srca_cb);

    geom_x.pop_front(1);
    geom_y.pop_front(1);
    if constexpr (FROM_OFFSETS) {
        offset_x.pop_front(1);
        offset_y.pop_front(1);
    }

    frac_x.wait_front(1);
    frac_y.wait_front(1);

    corner_weight(cb.frac_x, cb.frac_y, cb.attn, cb.scalar, /*invert_x=*/true, /*invert_y=*/true, srca_cb);   // NW
    corner_weight(cb.frac_x, cb.frac_y, cb.attn, cb.scalar, /*invert_x=*/false, /*invert_y=*/true, srca_cb);  // NE
    corner_weight(cb.frac_x, cb.frac_y, cb.attn, cb.scalar, /*invert_x=*/true, /*invert_y=*/false, srca_cb);  // SW
    corner_weight(cb.frac_x, cb.frac_y, cb.attn, cb.scalar, /*invert_x=*/false, /*invert_y=*/false, srca_cb); // SE

    frac_x.pop_front(1);
    frac_y.pop_front(1);
    attn.pop_front(1);
}

}  // namespace fused_msda_geometry
