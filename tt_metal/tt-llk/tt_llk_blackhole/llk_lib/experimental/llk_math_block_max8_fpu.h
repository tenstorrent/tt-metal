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
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu/experimental/ckernel_sfpu_block_max8.h"

namespace ckernel
{

namespace block_max8_gmpool
{

// FPU moves here use an all-zero address modifier; ADDR_MOD_3 avoids the slots datacopy programs.
constexpr std::uint32_t ADDR_MOD = ADDR_MOD_3;
// TRNSPSRCB transposes SrcB rows 16..31 only.
constexpr std::uint32_t SRCB_T = 16;

/**
 * @brief First physical DST row of the left-face rows 4p..4p+3 of the tile.
 * @param p Output row, 0 through 7.
 */
constexpr std::uint32_t left_row(std::uint32_t p)
{
    return 32 * (p / 4) + 4 * (p % 4);
}

/**
 * @brief Fill SrcB rows 16 + 4u .. 16 + 4u + 3 with the face rows of logical row 4p + u.
 *
 * Blocks 0/1 live in the left face row and 2/3 in the right face row, so each
 * face row is copied twice. After TRNSPSRCB, lane 4u + b of SrcB row 16 + c is
 * column c of that copy: lanes are already in output block order.
 */
template <std::uint32_t P, std::uint32_t U>
inline void load_logical_row()
{
    constexpr std::uint32_t left = left_row(P) + U;
    TTI_MOVD2B(0, SRCB_T + 4 * U + 0, ADDR_MOD, p_movd2b::MOV_1_ROW, left);
    TTI_MOVD2B(0, SRCB_T + 4 * U + 1, ADDR_MOD, p_movd2b::MOV_1_ROW, left);
    TTI_MOVD2B(0, SRCB_T + 4 * U + 2, ADDR_MOD, p_movd2b::MOV_1_ROW, left + FACE_R_DIM);
    TTI_MOVD2B(0, SRCB_T + 4 * U + 3, ADDR_MOD, p_movd2b::MOV_1_ROW, left + FACE_R_DIM);
}

/**
 * @brief Pool face columns [8 * HALF, 8 * HALF + 8) of the transposed rows into one DST row.
 *
 * SrcA rows 8..15 hold negative infinity from the unpacker clear, so the 16-row
 * GMPOOL is a max of eight. GMPOOL also maxes with its DST row, so that row is
 * first seeded with negative infinity from SrcA row 8. ZEROACC is avoided: on
 * Blackhole its one-row mode clears eight rows, and with MOVB2A it was seen to
 * shift the DST row counter across kernel runs. GMPOOL zeroes the three rows
 * below its target; those rows were already consumed by load_logical_row.
 */
template <std::uint32_t HALF, std::uint32_t DST_ROW>
inline void pool_half()
{
    TTI_MOVA2D(0, 8, ADDR_MOD, p_mova2d::MOV_1_ROW, DST_ROW);
    TTI_MOVB2A(0, ADDR_MOD, p_movb2a::MOV_4_ROWS, SRCB_T + 8 * HALF + 0);
    TTI_MOVB2A(4, ADDR_MOD, p_movb2a::MOV_4_ROWS, SRCB_T + 8 * HALF + 4);
    TTI_GATESRCRST(0b1, 0b1);
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD, p_gpool::INDEX_DIS, DST_ROW);
}

/**
 * @brief Write even-block maxima for output row P at left_row(P) and odd-block maxima 16 rows below.
 */
template <std::uint32_t P>
inline void pool_output_row()
{
    load_logical_row<P, 0>();
    load_logical_row<P, 1>();
    load_logical_row<P, 2>();
    load_logical_row<P, 3>();
    TTI_GATESRCRST(0b1, 0b1);
    TTI_TRNSPSRCB;
    TTI_GATESRCRST(0b1, 0b1);
    pool_half<0, left_row(P)>();
    pool_half<1, left_row(P) + FACE_R_DIM>();
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
    (TTI_MOVD2B(0, SRCB_T + P, ADDR_MOD, p_movd2b::MOV_1_ROW, left_row(P)), ...);
    (TTI_MOVD2B(0, SRCB_T + 8 + P, ADDR_MOD, p_movd2b::MOV_1_ROW, left_row(P) + FACE_R_DIM), ...);
}

} // namespace block_max8_gmpool

/**
 * @brief Initialize SFPU state for masking and the final merge, plus the zero FPU address modifier.
 * @note Call before @ref _llk_math_block_max8_fpu_. Pair each call with @ref _llk_unpack_block_max8_fpu_.
 */
inline void _llk_math_block_max8_fpu_init_()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(block_max8_gmpool::ADDR_MOD);
}

/**
 * @brief Replace one BF16 DST tile with 128 block maxima in its first eight physical rows, using GMPOOL.
 *
 * Masking (partial tiles only) runs on the SFPU before pooling. The pooled rows
 * are gathered through SrcB, SrcA/SrcB are released, and a final SFPU pass takes
 * the odd lanes from the odd-block rows, since GMPOOL writes whole rows only.
 * @param dst_index Tile index in the acquired BF16 DST half.
 * @param valid_scores Length of the valid row-major prefix, from zero through 1024.
 * @note Call @ref _llk_math_block_max8_fpu_init_ first. The unpack thread must
 *       issue @ref _llk_unpack_block_max8_fpu_ for each call.
 */
inline void _llk_math_block_max8_fpu_(std::uint32_t dst_index, std::uint32_t valid_scores)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if (valid_scores < TILE_R_DIM * TILE_C_DIM)
    {
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
        sfpu::_calculate_block_max8_mask_(valid_scores);
    }
    // MOVD2B does not wait for SrcB, and MOVB2A does not wait for SrcA.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    // Keep exponent-zero BF16 bit patterns intact through the moves, as transpose_dest does.
    math::_configure_preserve_zero_flag_state_();

    block_max8_gmpool::pool_all(std::make_integer_sequence<std::uint32_t, 8> {});

    block_max8_gmpool::gather_all(std::make_integer_sequence<std::uint32_t, 8> {});
    TTI_GATESRCRST(0b1, 0b1);
    TTI_MOVB2D(0, block_max8_gmpool::SRCB_T + 0, block_max8_gmpool::ADDR_MOD, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, block_max8_gmpool::SRCB_T + 4, block_max8_gmpool::ADDR_MOD, p_movb2d::MOV_4_ROWS, 4);
    TTI_MOVB2D(0, block_max8_gmpool::SRCB_T + 8, block_max8_gmpool::ADDR_MOD, p_movb2d::MOV_4_ROWS, 8);
    TTI_MOVB2D(0, block_max8_gmpool::SRCB_T + 12, block_max8_gmpool::ADDR_MOD, p_movb2d::MOV_4_ROWS, 12);
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);

    // Rows 0..7 hold even-block maxima in every lane; odd lanes (address + 2) come from rows 8..15.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    TTI_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, 8 + 2);
    TTI_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, 0 + 2);
    TTI_SFPLOAD(p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_FP16B, ADDR_MOD_7, 12 + 2);
    TTI_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_FP16B, ADDR_MOD_7, 4 + 2);
    math::clear_dst_reg_addr();
}

} // namespace ckernel
