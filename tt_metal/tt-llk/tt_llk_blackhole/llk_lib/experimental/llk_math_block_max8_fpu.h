// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// FPU (GMPOOL) variant of the block-max-of-eight pooling in
// sfpu/experimental/ckernel_sfpu_block_max8.h, with the same contract except
// that a block whose maximum is -0.0 or a denormal returns +0.0 (GMPOOL
// flushes exponent-zero inputs).
//
// Output row p, lane 4u + b holds block b of logical row 4p + u. For each p:
//   1. MOVD2B copies the face rows of logical rows 4p..4p+3 into SrcB rows
//      16..31, each face row twice: lanes 4u+0/1 get the left face row
//      (blocks 0, 1), lanes 4u+2/3 the right face row (blocks 2, 3).
//   2. TRNSPSRCB makes SrcB row 16 + c hold column c of every copy, so rows
//      16..23 hold the first block of each copy and rows 24..31 the second.
//   3. Two GMPOOLs reduce those eight rows: the even-block row is correct in
//      lanes b = 0, 2 and the odd-block row in lanes b = 1, 3.
// Finally MOVD2B/MOVB2D gather the 16 pooled rows into DST rows 0..15 and four
// SFPU instructions copy the odd lanes of rows 8..15 over rows 0..7.

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu/experimental/ckernel_sfpu_block_max8.h"

namespace ckernel
{

namespace block_max8_gmpool
{

// All-zero address modifier for the FPU instructions below: the SFPU init programs ADDR_MOD_7 to zero.
constexpr std::uint32_t ADDR_MOD = ADDR_MOD_7;
// Scores per block: each GMPOOL reduces this many transposed SrcB rows.
constexpr std::uint32_t BLOCK_WIDTH = 8;
// Output rows per tile. The odd-block maxima are gathered ODD_ROWS rows below the even-block maxima.
constexpr std::uint32_t OUTPUT_ROWS = 8;
constexpr std::uint32_t ODD_ROWS    = OUTPUT_ROWS;
// First SrcA row of the negative-infinity padding (rows 8..15) that the unpacker clears.
constexpr std::uint32_t SRCA_NEG_INF_ROW = 8;

/**
 * @brief Fill SrcB rows 16 + 4u .. 16 + 4u + 3 with the face rows of logical row 4p + u.
 *
 * Blocks 0/1 live in the left face row and 2/3 in the right face row, so each
 * face row is copied twice. After TRNSPSRCB, lane 4u + b of SrcB row 16 + c is
 * column c of that copy: lanes are already in output block order.
 *
 * @tparam P: Output row, 0 through 7.
 * @tparam U: Logical row within the group of four that output row P covers, 0 through 3.
 */
template <std::uint32_t P, std::uint32_t U>
inline void load_logical_row()
{
    constexpr std::uint32_t left = sfpu::block_max8::face_row_address(P) + U;
    constexpr std::uint32_t srcb = p_movd2b::SRC_ROW16_OFFSET + 4 * U;
    TTI_MOVD2B(p_mov::DEST_NORM, srcb + 0, ADDR_MOD, p_movd2b::MOV_1_ROW, left);
    TTI_MOVD2B(p_mov::DEST_NORM, srcb + 1, ADDR_MOD, p_movd2b::MOV_1_ROW, left);
    TTI_MOVD2B(p_mov::DEST_NORM, srcb + 2, ADDR_MOD, p_movd2b::MOV_1_ROW, left + FACE_R_DIM);
    TTI_MOVD2B(p_mov::DEST_NORM, srcb + 3, ADDR_MOD, p_movd2b::MOV_1_ROW, left + FACE_R_DIM);
}

/**
 * @brief Pool face columns [8 * HALF, 8 * HALF + 8) of the transposed rows into one DST row.
 *
 * SrcA rows 8..15 hold negative infinity from the unpacker clear, so the 16-row
 * GMPOOL is a max of eight. GMPOOL also maxes with its DST row, so that row is
 * first seeded with negative infinity from SrcA row 8. ZEROACC is avoided: on
 * Blackhole its one-row mode clears from the target's 4-row group to the end
 * of its 8-row block (measured), which would also clear input rows that later
 * output rows still read. GMPOOL zeroes the three rows below its target; those
 * rows were already consumed by load_logical_row.
 *
 * @tparam HALF: 0 pools the first block of each copy (SrcB rows 16..23), 1 the second (rows 24..31).
 * @tparam DST_ROW: DST row that receives the maxima.
 */
template <std::uint32_t HALF, std::uint32_t DST_ROW>
inline void pool_half()
{
    constexpr std::uint32_t srcb = p_movb2a::SRCB_ROW16_OFFSET + BLOCK_WIDTH * HALF;
    TTI_MOVA2D(p_mov::DEST_NORM, SRCA_NEG_INF_ROW, ADDR_MOD, p_mova2d::MOV_1_ROW, DST_ROW);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD, p_movb2a::MOV_4_ROWS, srcb + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD, p_movb2a::MOV_4_ROWS, srcb + 4);
    TTI_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/);
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD, p_gpool::INDEX_DIS, DST_ROW);
}

/**
 * @brief Write even-block maxima for output row P at its left-face DST row and odd-block maxima 16 rows below.
 *
 * @tparam P: Output row, 0 through 7.
 */
template <std::uint32_t P>
inline void pool_output_row()
{
    constexpr std::uint32_t left = sfpu::block_max8::face_row_address(P);
    load_logical_row<P, 0 /*U*/>();
    load_logical_row<P, 1 /*U*/>();
    load_logical_row<P, 2 /*U*/>();
    load_logical_row<P, 3 /*U*/>();
    TTI_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/);
    TTI_TRNSPSRCB;
    TTI_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/);
    pool_half<0 /*HALF*/, left>();
    pool_half<1 /*HALF*/, left + FACE_R_DIM>();
}

// Expand over output rows at compile time so every DST row address is an instruction immediate.
template <std::uint32_t... P>
inline void pool_all(std::integer_sequence<std::uint32_t, P...>)
{
    (pool_output_row<P>(), ...);
}

template <std::uint32_t... P>
inline void gather_all(std::integer_sequence<std::uint32_t, P...>)
{
    (TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + P, ADDR_MOD, p_movd2b::MOV_1_ROW, sfpu::block_max8::face_row_address(P)), ...);
    (TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET + ODD_ROWS + P, ADDR_MOD, p_movd2b::MOV_1_ROW, sfpu::block_max8::face_row_address(P) + FACE_R_DIM),
     ...);
}

} // namespace block_max8_gmpool

/**
 * @brief Initialize SFPU state for masking and the final merge, including the all-zero ADDR_MOD_7 the FPU moves also use.
 *
 * @note Call before @ref _llk_math_block_max8_fpu_.
 */
inline void _llk_math_block_max8_fpu_init_()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
}

/**
 * @brief Replace one BF16 DST tile with 128 block maxima in its first eight physical rows, using GMPOOL.
 *
 * Masking (partial tiles only) runs on the SFPU before pooling. The pooled rows
 * are gathered through SrcB, SrcA/SrcB are released, and a final SFPU pass takes
 * the odd lanes from the odd-block rows, since GMPOOL writes whole rows only.
 *
 * @param dst_index: Tile index in the acquired BF16 DST half.
 * @param valid_scores: Length of the valid row-major prefix, from zero through 1024.
 * @note Call @ref _llk_math_block_max8_fpu_init_ first. The unpack thread must
 *       issue @ref _llk_unpack_block_max8_fpu_ for each call.
 * @note SrcA and SrcB must be configured as Float16_b. The moves and GMPOOL read
 *       them in their configured format, and Float16 gives wrong maxima.
 */
inline void _llk_math_block_max8_fpu_(std::uint32_t dst_index, std::uint32_t valid_scores)
{
    using namespace block_max8_gmpool;

    LLK_ASSERT(
        math::src_zero_flag_srca_fmt == to_underlying(DataFormat::Float16_b) && math::src_zero_flag_srcb_fmt == to_underlying(DataFormat::Float16_b),
        "FPU block max8 requires Float16_b SrcA and SrcB");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if (valid_scores < TILE_R_DIM * TILE_C_DIM)
    {
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
        sfpu::_calculate_block_max8_mask_(valid_scores);
    }
    // MOVD2B does not wait for SrcB, and MOVB2A does not wait for SrcA. Drain math first so that a
    // pending SETRWC(CLR_AB) bank flip lands before the bank-validity check.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    // Keep exponent-zero BF16 bit patterns intact through the moves, as transpose_dest does. Save the
    // caller's flag first: a following matmul asserts the operand default, while a copy_init caller keeps it set.
    const std::uint32_t saved_zero_flag = math::src_zero_flag_hw;
    math::_configure_preserve_zero_flag_state_();

    pool_all(std::make_integer_sequence<std::uint32_t, OUTPUT_ROWS> {});

    gather_all(std::make_integer_sequence<std::uint32_t, OUTPUT_ROWS> {});
    TTI_GATESRCRST(1 /*reset_srcb_gate_control*/, 1 /*reset_srca_gate_control*/);
    TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 0, ADDR_MOD, p_movb2d::MOV_4_ROWS, 0 /*dst*/);
    TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD, p_movb2d::MOV_4_ROWS, 4 /*dst*/);
    TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD, p_movb2d::MOV_4_ROWS, 8 /*dst*/);
    TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD, p_movb2d::MOV_4_ROWS, 12 /*dst*/);
    // SET_ABD also returns the DST row counter to zero for the SFPU pass below.
    TTI_SETRWC(p_setrwc::CLR_AB, 0 /*rwc_cr*/, 0 /*rwc_d*/, 0 /*rwc_b*/, 0 /*rwc_a*/, p_setrwc::SET_ABD);
    // Restore the caller's flag so the next op needs no re-init. An unknown saved value stays unknown.
    if (saved_zero_flag == 0xff)
    {
        math::_invalidate_src_zero_flag_state_();
    }
    else
    {
        math::_configure_src_zero_flag_(saved_zero_flag != 0);
    }

    // Rows 0..7 hold even-block maxima in every lane; odd lanes (address + 2) come from rows 8..15.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    TTI_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, ODD_ROWS + 2);
    TTI_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, 0 + 2);
    TTI_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, ODD_ROWS + 4 + 2);
    TTI_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, 4 + 2);
}

} // namespace ckernel
