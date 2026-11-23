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

// Reduce tail of mul_reduce_scalar. Each DEST tile is pooled into row 0 of tile 0, and that row is
// pooled once more into datum 0. The scaler in SrcB row 0 applies to both pools.

/**
 * @brief Fidelity phase counts of the two pool stages.
 */
struct MulReduceScalarPhases
{
    std::uint32_t column = 1;
    std::uint32_t final  = 1;
};

inline MulReduceScalarPhases mul_reduce_scalar_phases {};

/**
 * @brief Shape of the reduced tiles, used by the moves into SrcA.
 */
inline ckernel::TensorShape mul_reduce_scalar_shape = ckernel::DEFAULT_TENSOR_SHAPE;

/**
 * @brief Pool one 16x16 face over @p num_phases fidelity phases, ending with @p last_addr_mod.
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
 * The pools read operands moved out of DEST, so the fidelity follows the DEST format.
 *
 * @tparam EN_32BIT_DEST: Set when DEST holds 32-bit datums.
 * @tparam MATH_FIDELITY_TYPE: Requested fidelity for both pool stages.
 * @param src_format: Unpacked source format of the reduced operand.
 * @param tensor_shape: Shape of the reduced tiles.
 * @note On the unpack thread, pair with @ref _llk_unpack_mul_reduce_scalar_switch_to_reduce_ (T0).
 * @note Call after the multiply or copy init, whose DEST tile stride the tail reuses.
 */
template <bool EN_32BIT_DEST, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_mul_reduce_scalar_init_(const DataFormat src_format, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    // Keep the DEST tile stride from the multiply or copy init. They differ for small tiles.
    mul_reduce_scalar_shape = tensor_shape;

    const DataFormat dest_src = dest_src_format<EN_32BIT_DEST>(src_format);
    const auto fidelity       = math_fidelity_schedule<MATH_FIDELITY_TYPE>(dest_src, dest_src);

    mul_reduce_scalar_phases.column = fidelity.phase_count;
    mul_reduce_scalar_phases.final  = fidelity.phase_count;

    // Last phase of a pool, and every move
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    // Last phase of a column pool, which also steps SrcA to the next face
    addr_mod_t {.srca = {.incr = FACE_R_DIM}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    // Every phase before the last
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_2);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Move DEST tile @p tile_idx into SrcA, or row 0 of it into SrcB.
 *
 * Each face goes to its own 16 rows of SrcA. Rows a short face leaves untouched stay zero.
 *
 * @tparam BINARY_REUSE_DEST: DEST_TO_SRCA or DEST_TO_SRCB.
 * @param tile_idx: DEST tile to move.
 * @note Needs the cleared bank from @ref _llk_unpack_mul_reduce_scalar_switch_to_reduce_.
 */
template <EltwiseBinaryReuseDestType BINARY_REUSE_DEST>
inline void _llk_math_mul_reduce_scalar_move_dest_to_src_(const std::uint32_t tile_idx)
{
    static_assert(
        BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCA || BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCB,
        "move_dest_to_src needs DEST_TO_SRCA or DEST_TO_SRCB");

    // The column pool leaves the SrcA counter past the tile, and moves are relative to it.
    _set_dst_write_addr_by_rows_(tile_idx);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD);

    if constexpr (BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // Wait for earlier pools and the SFPU to finish, and for math to own the SrcA bank.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCA_VLD);

        const ckernel::TensorShape& shape  = mul_reduce_scalar_shape;
        const std::uint32_t face_dest_rows = padded_face_rows(shape);
        const bool whole_fpu_groups        = (shape.face_r_dim % ELTWISE_MATH_ROWS) == 0;
        const std::uint32_t rows_per_move  = whole_fpu_groups ? ELTWISE_MATH_ROWS : 1;
        const std::uint32_t move_mode      = whole_fpu_groups ? FPU_MOV_ROWS : p_movd2a::MOV_1_ROW;
        for (std::uint32_t face = 0; face < shape.total_num_faces(); face++)
        {
            for (std::uint32_t row = 0; row < shape.face_r_dim; row += rows_per_move)
            {
                TT_MOVD2A(0 /*dest_32b_lo*/, face * FACE_R_DIM + row /*src*/, ADDR_MOD_0, move_mode, face * face_dest_rows + row /*dst*/);
            }
        }
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCB_VLD);
        TTI_MOVD2B(0 /*dest_32b_lo*/, 0 /*src*/, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0 /*transpose*/, 0 /*dst*/);
    }
}

/**
 * @brief Add the column sums of the tile in SrcA to row 0 of DEST tile @p dst_index.
 *
 * @param dst_index: DEST tile that accumulates the column sums (normally 0).
 * @param tensor_shape: Shape of the reduced tile.
 */
inline void _llk_math_mul_reduce_column_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    // The pool adds into the row the SFPU just zeroed.
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    _set_dst_write_addr_by_rows_(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD_F);

    for (std::uint32_t face = 0; face < tensor_shape.total_num_faces(); face++)
    {
        mul_reduce_scalar_pool_(mul_reduce_scalar_phases.column, ADDR_MOD_1);
    }
}

/**
 * @brief Sum row 0 of DEST tile 0 into datum 0, by transposing it through SrcB and pooling again.
 *
 * @tparam EN_32BIT_DEST: Set when DEST holds 32-bit datums.
 */
template <bool EN_32BIT_DEST>
inline void _llk_math_mul_reduce_scalar_()
{
    // Row 0 must be fully written before it is read back.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCB_VLD);
    _set_dst_write_addr_by_rows_(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD_F);

    TTI_MOVD2B(0 /*dest_32b_lo*/, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1 /*transpose*/, 0 /*dst*/);
#pragma GCC unroll 4
    for (const auto row : fpu_row_offsets<FACE_R_DIM>())
    {
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + row, ADDR_MOD_0, FPU_MOV_ROWS, p_movb2a::SRCB_ROW32_OFFSET + row);
    }

    // The final pool adds into row 0, so clear it first.
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, EN_32BIT_DEST, 0 /*clear_zero_flags*/, ADDR_MOD_0, 0 /*where*/);

    mul_reduce_scalar_pool_(mul_reduce_scalar_phases.final, ADDR_MOD_0);
}

/**
 * @brief Release the SrcA and SrcB banks held by the reduce tail.
 */
inline void _llk_math_mul_reduce_scalar_clear_dvalid_()
{
    TTI_CLEARDVALID(p_cleardvalid::CLR_SRCAB_VLD, 0, 0, 0, 0, 0);
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
