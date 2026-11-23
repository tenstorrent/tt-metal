// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/*************************************************************************
 * LLK MUL REDUCE SCALAR (Quasar) - reduce tail of the fused mul + scalar reduce
 *
 * The multiply phase is the regular ELWMUL into one DEST tile per input tile. The reduce tail then
 * runs entirely on the math thread: each product tile is moved from DEST into SrcA, column-pooled
 * into DEST tile 0 row 0 (all faces pool into the same 16-wide row), and the final row is
 * transposed through SrcB and pooled once more into datum 0. The scaler sits in SrcB row 0 for
 * both pool stages, so it multiplies the result twice.
 *************************************************************************/

/**
 * @brief Fidelity phase counts for the two pool stages, resolved from the source formats at init.
 */
struct MulReduceScalarPhases
{
    std::uint32_t column = 1;
    std::uint32_t final  = 1;
};

inline MulReduceScalarPhases mul_reduce_scalar_phases {};

/**
 * @brief Issue one pool over a 16x16 face with the given number of fidelity phases.
 *
 * The phases before the last advance the fidelity counter with ADDR_MOD_2; the last uses
 * @p last_addr_mod, which clears it.
 */
inline void mul_reduce_scalar_pool_(const std::uint32_t num_phases, const std::uint8_t last_addr_mod)
{
    for (std::uint32_t phase = 1; phase < num_phases; phase++)
    {
        TTI_GAPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0 /*dst*/);
    }
    TT_GAPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, last_addr_mod, p_gpool::INDEX_DIS, 0 /*dst*/);
}

/**
 * @brief Initialize the reduce tail of mul_reduce_scalar.
 *
 * Programs the address modifiers and resolves the fidelity phase counts. Both pool stages read
 * operands that were moved out of DEST, so the schedule uses the DEST-derived source format.
 *
 * @tparam EN_32BIT_DEST: Set when DEST holds 32-bit datums.
 * @tparam MATH_FIDELITY_TYPE: Requested fidelity for both pool stages.
 * @param src_format: Unpacked source format of the reduced operand.
 * @note On the unpack thread, pair with @ref _llk_unpack_mul_reduce_scalar_switch_to_reduce_ (T0).
 */
template <bool EN_32BIT_DEST, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_mul_reduce_scalar_init_(const DataFormat src_format)
{
    const DataFormat dest_src = dest_src_format<EN_32BIT_DEST>(src_format);
    const auto fidelity       = math_fidelity_schedule<MATH_FIDELITY_TYPE>(dest_src, dest_src);

    mul_reduce_scalar_phases.column = fidelity.phase_count;
    mul_reduce_scalar_phases.final  = fidelity.phase_count;

    // ADDR_MOD_0: no counter moves, fidelity cleared (last phase of a pool, and every move).
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    // ADDR_MOD_1: last phase of a column pool; steps SrcA to the next face.
    addr_mod_t {.srca = {.incr = FACE_R_DIM}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    // ADDR_MOD_2: every phase before the last.
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_2);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Move one DEST tile into SrcA (DEST_TO_SRCA) or the first face of DEST tile into SrcB (DEST_TO_SRCB).
 *
 * SrcA receives the whole tile so the column pool can walk its faces. SrcB only needs face 0, whose
 * row 0 carries the scaler for both pool stages.
 *
 * @tparam BINARY_REUSE_DEST: DEST_TO_SRCA or DEST_TO_SRCB.
 * @param tile_idx: DEST tile to move.
 * @note The source register must hold a dummy dvalid from @ref _llk_unpack_mul_reduce_scalar_switch_to_reduce_.
 */
template <EltwiseBinaryReuseDestType BINARY_REUSE_DEST>
inline void _llk_math_mul_reduce_scalar_move_dest_to_src_(const std::uint32_t tile_idx)
{
    static_assert(
        BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCA || BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCB,
        "move_dest_to_src needs DEST_TO_SRCA or DEST_TO_SRCB");

    // The source row of a move is offset by the SrcA/SrcB counter, which the column pool leaves at
    // the end of the tile, so reset it along with the DEST counter.
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD);

    if constexpr (BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // Drain the FPU (earlier pools still read SrcA) and the SFPU (it may have just written DEST),
        // then wait until math owns the SrcA bank.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCA_VLD);
#pragma GCC unroll 8
        for (const auto row : fpu_row_offsets<FACE_R_DIM * NUM_FACES>())
        {
            TTI_MOVD2A(0 /*dest_32b_lo*/, row /*src*/, ADDR_MOD_0, FPU_MOV_ROWS, row /*dst*/);
        }
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCB_VLD);
#pragma GCC unroll 4
        for (const auto row : fpu_row_offsets<FACE_R_DIM>())
        {
            TTI_MOVD2B(0 /*dest_32b_lo*/, row /*src*/, ADDR_MOD_0, FPU_MOV_ROWS, 0 /*transpose*/, row /*dst*/);
        }
    }
}

/**
 * @brief Column-pool the tile in SrcA into row 0 of DEST tile @p dst_index, accumulating.
 *
 * Every face pools into the same 16-wide row, so the row ends up holding per-column sums over the
 * whole tile. Call once per product tile after moving it into SrcA.
 *
 * @param dst_index: DEST tile that accumulates the column sums (normally 0).
 * @param tensor_shape: Shape of the reduced tile.
 */
inline void _llk_math_mul_reduce_column_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(tensor_shape.face_r_dim == FACE_R_DIM, "mul_reduce_scalar supports 16-row faces only");

    // The SFPU may have just zeroed the accumulator row; the pool adds into it.
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    _set_dst_write_addr_<DstTileShape::Tile32x32>(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD_F);

    for (std::uint32_t face = 0; face < tensor_shape.total_num_faces(); face++)
    {
        mul_reduce_scalar_pool_(mul_reduce_scalar_phases.column, ADDR_MOD_1);
    }
}

/**
 * @brief Collapse the column sums in DEST tile 0 row 0 into datum 0.
 *
 * The row is transposed into SrcB rows 32..47, copied to SrcA rows 0..15, and pooled once more.
 * SrcB rows 0..15 still hold the scaler.
 *
 * @tparam EN_32BIT_DEST: Set when DEST holds 32-bit datums.
 */
template <bool EN_32BIT_DEST>
inline void _llk_math_mul_reduce_scalar_()
{
    // Earlier pools must have written row 0 before it is read back.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCB_VLD);
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD_F);

    TTI_MOVD2B(0 /*dest_32b_lo*/, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1 /*transpose*/, 0 /*dst*/);
#pragma GCC unroll 4
    for (const auto row : fpu_row_offsets<FACE_R_DIM>())
    {
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + row, ADDR_MOD_0, FPU_MOV_ROWS, p_movb2a::SRCB_ROW32_OFFSET + row);
    }

    // The final pool accumulates, so clear the row it lands in.
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, EN_32BIT_DEST, 0 /*clear_zero_flags*/, ADDR_MOD_0, 0 /*where*/);

    mul_reduce_scalar_pool_(mul_reduce_scalar_phases.final, ADDR_MOD_0);
}

/**
 * @brief Release the SrcA/SrcB banks the reduce tail held and reset the counters.
 */
inline void _llk_math_mul_reduce_scalar_clear_dvalid_()
{
    TTI_CLEARDVALID(p_cleardvalid::CLR_SRCAB_VLD, 0, 0, 0, 0, 0);
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
