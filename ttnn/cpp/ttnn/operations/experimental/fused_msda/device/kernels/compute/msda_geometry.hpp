// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sampling geometry for fused multi-scale deformable attention, on the SFPU
// (and, with the tiled geometry, FPU matmuls).
//
// Why the geometry runs here
// --------------------------
// It is per-point float work over 32 query rows: vector work. The dataflow RISC
// has no FPU, so evaluating it there costs ~140 cycles per operation in
// soft-float emulation. The SFPU evaluates the same expression across the tile.
//
// What one point computes
// -----------------------
// Per sampling point (one (level, point) pair) and a block of up to 32 queries:
//
//   px   = primary * primary_scale + secondary * secondary_scale + bias
//   x0   = floor(px)
//   dx   = px - x0
//   w_c  = corner(dx, dy) * attn                        for the four corners
//   w_c  = 0 where corner c is out of bounds            (the gathers skip it)
//
// `primary` is the normalized x (V1) or the reference point's x (V2);
// `secondary` is the raw sampling offset and exists only for V2. The
// (align_corners, locations_in_grid_space, from_offsets) variants are folded
// into the three per-level constants on the host, so this kernel is branch-free
// on them — see fused_msda_program_factory.cpp::axis_constants.
//
// Queries occupy tile rows and only column 0 carries a scalar, which is what
// `mul_tiles_bcast<COL>` consumes in the reduction. Column 0 lives in faces 0
// and 2, so the whole solve runs on those two faces only (VectorMode::C), and
// column 16 — the same SFPU lane, one face to the right — carries the second
// operand of each pair:
//
//   DST slot  in (col 0 | col 16)     out (col 0 | col 16)
//   0         primary x | secondary x  w_NW | corner word
//   1         primary y | secondary y  w_NE | y0
//   2         attn      | -            w_SW | -
//   3         -         | -            w_SE | -
//
// One copy per operand tile, one acquire window, five packs. The fractions never
// leave DST.
//
// The gathers get the corner word (msda_corner_word.hpp): the bounds tests and
// the NW corner's index within its level, built here from the floored corner and
// the level's extent. Each gather only adds the per-corner step and the level's
// base.
//
// fp32_dest_acc_en is required, not an optimization. px reaches the feature
// map's extent, and bf16's ulp at 200 is 1.0 — in a 16-bit destination
// floor(px) on the larger levels rounds to the wrong integer and the fraction
// it feeds collapses, degrading bilinear sampling to nearest-neighbour.

#pragma once

#include <cstdint>

#include "api/compute/common.h"
// rounding_op_tile_init, and on the math thread ckernel::sfpu::_floor_body_ and
// _llk_math_eltwise_unary_sfpu_params_.
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_corner_word.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tiled_geometry.hpp"

namespace fused_msda_geometry {

// DST slots of the table above, named for what they hold on the way in and on
// the way out. Four is the entire budget a 32-bit destination allows under
// DstSync::Half (`dest_size = is_fp32_dest_acc_en ? 4 : 8` in
// api/compute/tilize.h).
constexpr uint32_t DST_X = 0;  // in: primary | secondary x; out: w_NW | corner word
constexpr uint32_t DST_Y = 1;  // in: primary | secondary y; out: w_NE | y0
constexpr uint32_t DST_ATTN = 2;
constexpr uint32_t DST_NW = DST_X;
constexpr uint32_t DST_NE = DST_Y;
constexpr uint32_t DST_SW = DST_ATTN;
constexpr uint32_t DST_SE = 3;
constexpr uint32_t DST_FP32_CAPACITY = 4;
static_assert(DST_SE < DST_FP32_CAPACITY, "geometry uses a DST slot a 32-bit destination does not have");

// The host-folded normalized -> pixel mapping for one axis of one level:
//   px = primary * primary_scale + secondary * secondary_scale + bias
// and the axis' extent (W or H), which bounds the floored corner.
// `secondary_scale` is unused when the op runs without sampling offsets. All
// four are fp32 bit patterns; must stay in step with `AxisConstants` in
// fused_msda_program_factory.cpp, which builds them.
struct AxisConstants {
    uint32_t primary_scale;
    uint32_t secondary_scale;
    uint32_t bias;
    uint32_t extent;
};

// The circular buffers one geometry pass moves through. Grouped rather than
// passed positionally: same-typed CB indices in a row let a transposition
// (geom_x for geom_y) compile cleanly and produce transposed sampling that PCC
// survives on symmetric shapes.
struct GeometryPipes {
    uint32_t geom_x;  // reader -> compute, primary x | secondary x
    uint32_t geom_y;
    uint32_t attn;    // reader -> compute
    uint32_t corner;  // compute -> reader, float32, the corner word in column 16
    uint32_t scalar;  // compute -> compute, consumed by the reduction
};

namespace sfpu {
// SFPU addressing in dst_reg units: one face is 8 rows of 32 lanes, one tile 32.
// Column 16 of a row is the same lane as column 0, one face to the right.
constexpr uint32_t FACE = 8;
constexpr uint32_t TILE = 32;
}  // namespace sfpu

// SFPI bodies, compiled for the math thread only and run through
// _llk_math_eltwise_unary_sfpu_params_ with VectorMode::C, which calls them once
// for face 0 and once for face 2 of the tile at DST slot 0. Each names the slots
// it touches itself.
#ifdef TRISC_MATH
namespace sfpu {

inline sfpi::vFloat as_vfloat(uint32_t bits) { return sfpi::vFloat(__builtin_bit_cast(float, bits)); }

// px over one face of `slot`, written over the primary operand.
template <bool FROM_OFFSETS, uint32_t SLOT>
inline void pixel(uint32_t primary_scale, uint32_t secondary_scale, uint32_t bias) {
    const sfpi::vFloat primary_k = as_vfloat(primary_scale);
    const sfpi::vFloat bias_k = as_vfloat(bias);
    [[maybe_unused]] const sfpi::vFloat secondary_k = as_vfloat(secondary_scale);
#pragma GCC unroll 0
    for (uint32_t d = 0; d < FACE; ++d) {
        sfpi::vFloat px = sfpi::vFloat(sfpi::dst_reg[SLOT * TILE]) * primary_k + bias_k;
        if constexpr (FROM_OFFSETS) {
            px = sfpi::vFloat(sfpi::dst_reg[SLOT * TILE + FACE]) * secondary_k + px;
        }
        sfpi::dst_reg[SLOT * TILE] = px;
        sfpi::dst_reg++;
    }
}

// floor of the face at dst_reg offset SRC into the face at DST. One
// _floor_body_ per iteration and nothing else live: it clobbers LREG0-3 behind
// the compiler's back, which is safe only in the shape _calculate_floor_ uses
// it.
template <uint32_t SRC, uint32_t DST>
inline void floor_to() {
#pragma GCC unroll 0
    for (uint32_t d = 0; d < FACE; ++d) {
        sfpi::dst_reg[DST] = ckernel::sfpu::_floor_body_(sfpi::dst_reg[SRC]);
        sfpi::dst_reg++;
    }
}

// The four attn-weighted bilinear coefficients, NW NE SW SE, into the left
// faces of slots 0..3. The template arguments are the dst_reg offsets of px,
// floor(px), py, floor(py) and attn; every input is read before the first
// output is written, so outputs may overwrite inputs. CLEAR_ATTN_RIGHT zeroes
// the face to the right of attn, which the caller then packs as part of a tile.
template <uint32_t PX, uint32_t X0, uint32_t PY, uint32_t Y0, uint32_t ATTN, bool CLEAR_ATTN_RIGHT = false>
inline void corner_weights() {
#pragma GCC unroll 0
    for (uint32_t d = 0; d < FACE; ++d) {
        const sfpi::vFloat dx = sfpi::vFloat(sfpi::dst_reg[PX]) - sfpi::dst_reg[X0];
        const sfpi::vFloat dy = sfpi::vFloat(sfpi::dst_reg[PY]) - sfpi::dst_reg[Y0];
        const sfpi::vFloat a = sfpi::dst_reg[ATTN];
        const sfpi::vFloat a_dy = a * dy;      // attn * dy
        const sfpi::vFloat a_1mdy = a - a_dy;  // attn * (1 - dy)
        const sfpi::vFloat ne = dx * a_1mdy;
        const sfpi::vFloat se = dx * a_dy;
        sfpi::dst_reg[DST_NW * TILE] = a_1mdy - ne;  // (1 - dx)(1 - dy) attn
        sfpi::dst_reg[DST_NE * TILE] = ne;           // dx (1 - dy) attn
        sfpi::dst_reg[DST_SW * TILE] = a_dy - se;    // (1 - dx) dy attn
        sfpi::dst_reg[DST_SE * TILE] = se;           // dx dy attn
        if constexpr (CLEAR_ATTN_RIGHT) {
            sfpi::dst_reg[ATTN + FACE] = 0.0f;
        }
        sfpi::dst_reg++;
    }
}

// The corner word (msda_corner_word.hpp) of floored corner (x, y) on a w x h
// level.
inline sfpi::vFloat corner_word(sfpi::vFloat x, sfpi::vFloat y, sfpi::vFloat w, sfpi::vFloat h) {
    namespace cw = fused_msda_corner_word;
    // SFPSWAP orders float32 by its sign-magnitude bits, so the clamps are
    // total: NaN and infinity land on an end of the range. The bounds tests
    // then compare finite integers. The SFPU compares against a constant by
    // subtracting it and testing the sign bit, which -0 has set; the tests run
    // on x + 2 and y + 2, whose differences are never -0.
    x = sfpi::max(sfpi::min(x, w), -2.0f);
    y = sfpi::max(sfpi::min(y, h), -2.0f);
    sfpi::vFloat word = (sfpi::max(y, -1.0f) + 1.0f) * w + (sfpi::max(x, -1.0f) + (1.0f + cw::BIAS));
    const sfpi::vFloat xs = x + 2.0f;
    const sfpi::vFloat ys = y + 2.0f;
    constexpr float UNIT = static_cast<float>(1u << cw::INDEX_BITS);
    v_if(xs >= 2.0f && xs < w + 2.0f) { word += UNIT * cw::FLAG_X0; }
    v_endif;
    v_if(xs >= 1.0f && xs < w + 1.0f) { word += UNIT * cw::FLAG_X1; }
    v_endif;
    v_if(ys >= 2.0f && ys < h + 2.0f) { word += UNIT * cw::FLAG_Y0; }
    v_endif;
    v_if(ys >= 1.0f && ys < h + 1.0f) { word += UNIT * cw::FLAG_Y1; }
    v_endif;
    return word;
}

// The corner word of the floored corner at X0 / Y0 into OUT, for one level.
template <uint32_t X0, uint32_t Y0, uint32_t OUT>
inline void corner_word_level(uint32_t width, uint32_t height) {
    const sfpi::vFloat w = as_vfloat(width);
    const sfpi::vFloat h = as_vfloat(height);
#pragma GCC unroll 0
    for (uint32_t d = 0; d < FACE; ++d) {
        sfpi::dst_reg[OUT] = corner_word(sfpi::dst_reg[X0], sfpi::dst_reg[Y0], w, h);
        sfpi::dst_reg++;
    }
}

// As above, with each lane's extent read from DST at W / H.
template <uint32_t X0, uint32_t Y0, uint32_t W, uint32_t H, uint32_t OUT>
inline void corner_word_lanes() {
#pragma GCC unroll 0
    for (uint32_t d = 0; d < FACE; ++d) {
        sfpi::dst_reg[OUT] = corner_word(sfpi::dst_reg[X0], sfpi::dst_reg[Y0], sfpi::dst_reg[W], sfpi::dst_reg[H]);
        sfpi::dst_reg++;
    }
}

// Zeroes the weight of every corner the corner word at WORD marks out of
// bounds. The gathers skip those corners and leave their input slots stale, so
// this zero is what keeps them out of the sum.
template <uint32_t WORD>
inline void mask_corner_weights() {
    namespace cw = fused_msda_corner_word;
    // Below 2^24 the word's bits are 0x4B000000 plus the integer, and the
    // exponent bits shifted down leave the low four bits to the flags.
    constexpr int32_t NW = cw::FLAG_X0 | cw::FLAG_Y0;
    constexpr int32_t NE = cw::FLAG_X1 | cw::FLAG_Y0;
    constexpr int32_t SW = cw::FLAG_X0 | cw::FLAG_Y1;
    constexpr int32_t SE = cw::FLAG_X1 | cw::FLAG_Y1;
#pragma GCC unroll 0
    for (uint32_t d = 0; d < FACE; ++d) {
        const sfpi::vInt bits = sfpi::dst_reg[WORD].mode<sfpi::DataLayout::I32>();
        const sfpi::vInt flags = (bits >> cw::INDEX_BITS) & 0xF;
        v_if((flags & NW) != NW) { sfpi::dst_reg[DST_NW * TILE] = 0.0f; }
        v_endif;
        v_if((flags & NE) != NE) { sfpi::dst_reg[DST_NE * TILE] = 0.0f; }
        v_endif;
        v_if((flags & SW) != SW) { sfpi::dst_reg[DST_SW * TILE] = 0.0f; }
        v_endif;
        v_if((flags & SE) != SE) { sfpi::dst_reg[DST_SE * TILE] = 0.0f; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
#endif

// Points the unpacker and datacopy at `new_cb`. For float32 geom_x/geom_y the
// program factory sets UnpackToDestFp32, so that copy bypasses SrcA and lands in
// DST with all 23 mantissa bits. `srca_cb` tracks what SrcA was last
// configured for so the reconfiguration is skipped when nothing changed; it must
// be seeded with whatever `compute_kernel_hw_startup` left SrcA on.
inline void copy_from(uint32_t new_cb, uint32_t& srca_cb) {
    reconfig_data_format_srca(srca_cb, new_cb);
    copy_init(new_cb);
    srca_cb = new_cb;
}

// Geometry for one sampling point: the corner word for the gathers, then the
// four corner scalars in the order NW, NE, SW, SE.
//
// That order is a contract with both gathers: the four scalar tiles pushed here
// are consumed against the corner slots of fused_msda_gather::gather_rows
// (row-major block [NW | NE | SW | SE]) and against the input-tile groups of
// the tile-scatter path (`dy = c >> 1; dx = c & 1` in
// fused_msda_reader_common.hpp). The orders must agree element for element or
// every sample is silently wrong.
//
// The corner tile is float32 and every other pack target bf16, so the packer is
// reconfigured around the corner pack and left on the scalar CB's format.
template <bool FROM_OFFSETS>
inline void point(const GeometryPipes& cb, const AxisConstants& kx, const AxisConstants& ky, uint32_t& srca_cb) {
    CircularBuffer geom_x(cb.geom_x);
    CircularBuffer geom_y(cb.geom_y);
    CircularBuffer attn(cb.attn);
    CircularBuffer corner_out(cb.corner);
    CircularBuffer scalar_out(cb.scalar);

    geom_x.wait_front(1);
    geom_y.wait_front(1);
    attn.wait_front(1);
    corner_out.reserve_back(1);
    scalar_out.reserve_back(4);

    tile_regs_acquire();

    // Every copy_tile first, then the maths. Interleaving an SFPU op between two
    // copies reconfigures the unpacker mid-window.
    copy_from(cb.geom_x, srca_cb);
    copy_tile(cb.geom_x, 0, DST_X);
    copy_from(cb.geom_y, srca_cb);
    copy_tile(cb.geom_y, 0, DST_Y);
    copy_from(cb.attn, srca_cb);
    copy_tile(cb.attn, 0, DST_ATTN);

    rounding_op_tile_init();
    // Every pass names its slots itself, so all of them start at slot 0.
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        sfpu::pixel<FROM_OFFSETS, DST_X>, 0, VectorMode::C, kx.primary_scale, kx.secondary_scale, kx.bias)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        sfpu::pixel<FROM_OFFSETS, DST_Y>, 0, VectorMode::C, ky.primary_scale, ky.secondary_scale, ky.bias)));
    constexpr uint32_t X = DST_X * sfpu::TILE;
    constexpr uint32_t Y = DST_Y * sfpu::TILE;
    constexpr uint32_t F = sfpu::FACE;
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::floor_to<X, X + F>, 0, VectorMode::C)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::floor_to<Y, Y + F>, 0, VectorMode::C)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        sfpu::corner_weights<X, X + F, Y, Y + F, DST_ATTN * sfpu::TILE>, 0, VectorMode::C)));
    // After the weights: the word overwrites the x0 they read.
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        sfpu::corner_word_level<X + F, Y + F, X + F>, 0, VectorMode::C, kx.extent, ky.extent)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::mask_corner_weights<X + F>, 0, VectorMode::C)));

    tile_regs_commit();
    tile_regs_wait();
    // The corner goes before the scalars: the reader waits on it, the
    // reduction does not need the scalars until after the gather.
    pack_reconfig_data_format(cb.scalar, cb.corner);
    pack_tile<true>(DST_X, cb.corner, 0);
    corner_out.push_back(1);
    pack_reconfig_data_format(cb.corner, cb.scalar);
    pack_tile<true>(DST_NW, cb.scalar, 0);
    pack_tile<true>(DST_NE, cb.scalar, 1);
    pack_tile<true>(DST_SW, cb.scalar, 2);
    pack_tile<true>(DST_SE, cb.scalar, 3);
    tile_regs_release();
    scalar_out.push_back(4);

    geom_x.pop_front(1);
    geom_y.pop_front(1);
    attn.pop_front(1);
}

// ---------------------------------------------------------------------------
// Tiled geometry (msda_tiled_geometry.hpp): one window per output block.
//
//   DST slot  in (cols 0..15 | 16..31)   out (cols 0..15 | 16..31)
//   0         px_j | py_j                 w_NW_j | py_j
//   1         attn_j | staging padding    w_NE_j | 0
//   2         y0_j | x0_j (floored)       w_SW_j | x0_j
//   3         H_j | W_j                   w_SE_j | corner word_j
//
// Slots 0 and 3 come out of matmuls that accumulate in the fp32 destination,
// so px | py never pass through a 19-bit source register. The staging padding
// is only copied, never multiplied: it may hold any bit pattern. Slots 2 and 3
// put y on the left: the weights overwrite the left faces, and the corner word
// must survive in the right face. Slot 3 goes to the reader as the block's
// corner tile, and all four to the weights CB, from which `extract` moves one
// point's column at a time to column 0. E_j's zeros cancel every other column
// of a weight tile; Blackhole's FPU multiplies inf and NaN by zero to zero.
// ---------------------------------------------------------------------------
struct BlockPipes {
    uint32_t rm;        // reader -> compute, row-major [offsets | attn] rows
    uint32_t operands;  // compute scratch, OFF_TILE and ATTN_TILE tilized
    uint32_t ref;       // reader -> compute, the reference tile
    uint32_t consts;    // reader -> compute once, never popped
    uint32_t weights;   // compute -> compute, w_NW w_NE w_SW w_SE
    uint32_t corner;    // compute -> reader, float32, corner word_j in column 16 + j
};

// Slot names for the table above. The outputs land where `corner_weights`
// writes them, so they alias DST_NW..DST_SE.
constexpr uint32_t BLK_PXPY = DST_NW;    // out: w_NW | py
constexpr uint32_t BLK_ATTN = DST_NE;    // out: w_NE | 0
constexpr uint32_t BLK_FLOOR = DST_SW;   // out: w_SW | x0
constexpr uint32_t BLK_EXTENT = DST_SE;  // out: w_SE | corner word

inline void block(const BlockPipes& cb, uint32_t& srca_cb) {
    namespace tg = fused_msda_tiled_geometry;
    CircularBuffer rm(cb.rm);
    CircularBuffer operands(cb.operands);
    CircularBuffer ref(cb.ref);
    CircularBuffer weights(cb.weights);
    CircularBuffer corner_out(cb.corner);

    rm.wait_front(tg::RM_TILES);
    operands.reserve_back(tg::RM_TILES);
    tilize_init(cb.rm, tg::RM_TILES, cb.operands);
    tilize_block(cb.rm, tg::RM_TILES, cb.operands);
    tilize_uninit(cb.rm, cb.operands);
    operands.push_back(tg::RM_TILES);
    rm.pop_front(tg::RM_TILES);

    operands.wait_front(tg::RM_TILES);
    ref.wait_front(1);
    weights.reserve_back(tg::WEIGHT_TILES);
    corner_out.reserve_back(1);

    tile_regs_acquire();

    // Every operand CB is bf16 32x32, so switching CBs needs no format
    // reconfiguration (see the compute config in the program factory).
    matmul_init(cb.operands, cb.consts);
    matmul_tiles(cb.operands, cb.consts, tg::OFF_TILE, tg::CONST_M_OFF, BLK_PXPY);
    matmul_init(cb.ref, cb.consts);
    matmul_tiles(cb.ref, cb.consts, 0, tg::CONST_M_REF, BLK_PXPY);
    matmul_tiles(cb.ref, cb.consts, 0, tg::CONST_M_EXTENT, BLK_EXTENT);
    srca_cb = cb.consts;  // matmul unpacks in1 to SrcA

    copy_from(cb.operands, srca_cb);
    copy_tile(cb.operands, tg::ATTN_TILE, BLK_ATTN);

    constexpr uint32_t T = sfpu::TILE;
    constexpr uint32_t F = sfpu::FACE;
    constexpr uint32_t PX = BLK_PXPY * T;
    constexpr uint32_t PY = BLK_PXPY * T + F;
    constexpr uint32_t X0 = BLK_FLOOR * T + F;
    constexpr uint32_t Y0 = BLK_FLOOR * T;
    constexpr uint32_t W = BLK_EXTENT * T + F;
    constexpr uint32_t H = BLK_EXTENT * T;
    rounding_op_tile_init();
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::floor_to<PX, X0>, 0, VectorMode::C)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::floor_to<PY, Y0>, 0, VectorMode::C)));
    // Before the weights, which overwrite H.
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::corner_word_lanes<X0, Y0, W, H, W>, 0, VectorMode::C)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        sfpu::corner_weights<PX, X0, PY, Y0, BLK_ATTN * T, true>, 0, VectorMode::C)));
    MATH((_llk_math_eltwise_unary_sfpu_params_(sfpu::mask_corner_weights<W>, 0, VectorMode::C)));

    tile_regs_commit();
    tile_regs_wait();
    // The corner tile first: the reader is waiting on it, the reduction is not.
    pack_reconfig_data_format(cb.weights, cb.corner);
    pack_tile<true>(BLK_EXTENT, cb.corner, 0);
    corner_out.push_back(1);
    pack_reconfig_data_format(cb.corner, cb.weights);
    pack_tile<true>(DST_NW, cb.weights, 0);
    pack_tile<true>(DST_NE, cb.weights, 1);
    pack_tile<true>(DST_SW, cb.weights, 2);
    pack_tile<true>(DST_SE, cb.weights, 3);
    tile_regs_release();
    weights.push_back(tg::WEIGHT_TILES);

    operands.pop_front(tg::RM_TILES);
    ref.pop_front(1);
}

// Point j's four corner scalars, NW NE SW SE, into the scalar CB in the order
// `point` produces them: column j of each weight tile, moved to column 0. The
// block's weight tiles stay at the front until its last point, which pops
// them; the next block's are pushed behind them meanwhile.
inline void extract(const BlockPipes& cb, uint32_t scalar_cb_index, uint32_t j, uint32_t& srca_cb) {
    namespace tg = fused_msda_tiled_geometry;
    CircularBuffer weights(cb.weights);
    CircularBuffer scalar_out(scalar_cb_index);
    weights.wait_front(tg::WEIGHT_TILES);
    scalar_out.reserve_back(4);
    matmul_init(cb.weights, cb.consts);
    srca_cb = cb.consts;
    tile_regs_acquire();
    for (uint32_t c = 0; c < 4; ++c) {
        matmul_tiles(cb.weights, cb.consts, c, tg::CONST_E_FIRST + j, c);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t c = 0; c < 4; ++c) {
        pack_tile<true>(c, scalar_cb_index, c);
    }
    tile_regs_release();
    scalar_out.push_back(4);
    if (j + 1 == tg::POINTS) {
        weights.pop_front(tg::WEIGHT_TILES);
    }
}

}  // namespace fused_msda_geometry
