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

// Reduce tail of mul_reduce_scalar. Each DEST tile is pooled into row 0 of tile 0, and that row is
// pooled once more into datum 0. The scaler in SrcB row 0 applies to both pools.
// Every call takes what it needs as arguments, so nothing is shared between TRISCs of different Neos.

/**
 * @brief Fidelity schedule of the pools for the formats in SrcA and SrcB.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline ckernel::math::MathFidelitySchedule mul_reduce_scalar_fidelity_(const DataFormat srca_format, const DataFormat srcb_format)
{
    return ckernel::math::math_fidelity_schedule<MATH_FIDELITY_TYPE>(srca_format, srcb_format);
}

/**
 * @brief Pool one 16x16 face over @p num_phases fidelity phases, ending with @p last_addr_mod.
 */
inline void mul_reduce_scalar_pool_(const std::uint32_t num_phases, const std::uint8_t last_addr_mod)
{
    using namespace ckernel;
    for (std::uint32_t phase = 1; phase < num_phases; phase++)
    {
        TTI_GAPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0 /*dst*/);
    }
    TT_GAPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, last_addr_mod, p_gpool::INDEX_DIS, 0 /*dst*/);
}

/**
 * @brief Initialize the reduce tail of mul_reduce_scalar.
 *
 * SrcA and SrcB keep the unpacker's formats, because MOVD2A and MOVD2B move only the top 16 bits of
 * a 32-bit DEST datum. The fidelity follows those formats.
 *
 * @tparam MATH_FIDELITY_TYPE: Requested fidelity for both pool stages.
 * @param srca_format: Unpacked format of the reduced operand, which SrcA holds.
 * @param srcb_format: Unpacked format of the second operand, which SrcB and the scaler use.
 * @param tensor_shape: Shape of the reduced tiles.
 * @note On the unpack thread, pair with @ref _llk_unpack_mul_reduce_scalar_switch_to_reduce_ (T0).
 * @note Call after the multiply or copy init, whose DEST tile stride the tail reuses. Pass the same
 *       formats and shape to the moves and pools that follow.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_mul_reduce_scalar_init_(const DataFormat srca_format, const DataFormat srcb_format, const ckernel::TensorShape& tensor_shape)
{
    using namespace ckernel;
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    const auto is_float_format = [](const DataFormat format)
    { return format == DataFormat::Float16 || format == DataFormat::Float16_b || format == DataFormat::Tf32 || format == DataFormat::Float32; };
    LLK_ASSERT(is_float_format(srca_format) && is_float_format(srcb_format), "mul_reduce_scalar pools float sources only");

    const auto fidelity = mul_reduce_scalar_fidelity_<MATH_FIDELITY_TYPE>(srca_format, srcb_format);

    // Last phase of a pool, and every move
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    // Last phase of a column pool, which also steps SrcA to the next face
    addr_mod_t {.srca = {.incr = FACE_R_DIM}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    // Every phase before the last
    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_2);

    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Move DEST tile @p tile_idx into SrcA, or row 0 of it into SrcB.
 *
 * Each face goes to its own 16 rows of SrcA. Rows a short face leaves untouched stay zero. Faces
 * shorter than an FPU row group pass through SrcB rows 32 and up, which the final pool reuses later.
 *
 * @tparam BINARY_REUSE_DEST: DEST_TO_SRCA or DEST_TO_SRCB.
 * @param tile_idx: DEST tile to move.
 * @param tensor_shape: Shape of the reduced tiles, as passed to the init.
 * @note Needs the cleared bank from @ref _llk_unpack_mul_reduce_scalar_switch_to_reduce_.
 */
template <ckernel::EltwiseBinaryReuseDestType BINARY_REUSE_DEST>
inline void _llk_math_mul_reduce_scalar_move_dest_to_src_(const std::uint32_t tile_idx, const ckernel::TensorShape& tensor_shape)
{
    using namespace ckernel;
    static_assert(
        BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCA || BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCB,
        "move_dest_to_src needs DEST_TO_SRCA or DEST_TO_SRCB");

    // The column pool leaves the SrcA counter past the tile, and moves are relative to it.
    math::_set_dst_write_addr_by_rows_(tile_idx);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD);

    if constexpr (BINARY_REUSE_DEST == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // Wait for earlier pools and the SFPU to finish, and for math to own the SrcA bank.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCA_VLD);

        const TensorShape& shape           = tensor_shape;
        const std::uint32_t face_dest_rows = padded_face_rows(shape);
        if (shape.face_r_dim % ELTWISE_MATH_ROWS == 0)
        {
            for (std::uint32_t face = 0; face < shape.total_num_faces(); face++)
            {
                for (std::uint32_t row = 0; row < shape.face_r_dim; row += ELTWISE_MATH_ROWS)
                {
                    TT_MOVD2A(0 /*dest_32b_lo*/, face * FACE_R_DIM + row /*src*/, ADDR_MOD_0, FPU_MOV_ROWS, face * face_dest_rows + row /*dst*/);
                }
            }
        }
        else
        {
            // ttsim has no one row MOVD2A, so a short face goes row by row into spare SrcB rows and then
            // moves to SrcA as one row group. The unwritten SrcB rows are still zero from the unpacker.
            LLK_ASSERT(shape.face_r_dim < ELTWISE_MATH_ROWS, "A short face must fit in one FPU row group");
            TTI_STALLWAIT(p_stall::STALL_MATH, 0 /*wait_res_idx_2*/, 0 /*wait_res_idx_1*/, p_stall::SRCB_VLD);
            for (std::uint32_t face = 0; face < shape.total_num_faces(); face++)
            {
                for (std::uint32_t row = 0; row < shape.face_r_dim; row++)
                {
                    TT_MOVD2B(
                        0 /*dest_32b_lo*/,
                        p_movd2b::SRC_ROW32_OFFSET + row /*src*/,
                        ADDR_MOD_0,
                        p_movd2b::MOV_1_ROW,
                        0 /*transpose*/,
                        face * face_dest_rows + row /*dst*/);
                }
                TT_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + face * FACE_R_DIM, ADDR_MOD_0, FPU_MOV_ROWS, p_movb2a::SRCB_ROW32_OFFSET);
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
 * @tparam MATH_FIDELITY_TYPE: Fidelity passed to the init.
 * @param dst_index: DEST tile that accumulates the column sums (normally 0).
 * @param srca_format: SrcA format passed to the init.
 * @param srcb_format: SrcB format passed to the init.
 * @param tensor_shape: Shape of the reduced tile.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_mul_reduce_column_(
    const std::uint32_t dst_index, const DataFormat srca_format, const DataFormat srcb_format, const ckernel::TensorShape& tensor_shape)
{
    using namespace ckernel;
    const std::uint32_t num_phases = mul_reduce_scalar_fidelity_<MATH_FIDELITY_TYPE>(srca_format, srcb_format).phase_count;

    // The pool adds into the row the SFPU just zeroed.
    TTI_STALLWAIT(p_stall::STALL_MATH, 0 /*wait_res_idx_2*/, 0 /*wait_res_idx_1*/, p_stall::WAIT_SFPU);
    math::_set_dst_write_addr_by_rows_(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD_F);

    for (std::uint32_t face = 0; face < tensor_shape.total_num_faces(); face++)
    {
        mul_reduce_scalar_pool_(num_phases, ADDR_MOD_1);
    }
}

/**
 * @brief Sum row 0 of DEST tile 0 into datum 0, by transposing it through SrcB and pooling again.
 *
 * @tparam MATH_FIDELITY_TYPE: Fidelity passed to the init.
 * @tparam EN_32BIT_DEST: Set when DEST holds 32-bit datums.
 * @param srca_format: SrcA format passed to the init.
 * @param srcb_format: SrcB format passed to the init.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool EN_32BIT_DEST>
inline void _llk_math_mul_reduce_scalar_(const DataFormat srca_format, const DataFormat srcb_format)
{
    using namespace ckernel;
    const std::uint32_t num_phases = mul_reduce_scalar_fidelity_<MATH_FIDELITY_TYPE>(srca_format, srcb_format).phase_count;

    // Row 0 must be fully written before it is read back.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCB_VLD);
    math::_set_dst_write_addr_by_rows_(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0 /*rwc_cr*/, 0 /*rwc_val*/, p_setrwc::SET_ABD_F);

    TTI_MOVD2B(0 /*dest_32b_lo*/, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1 /*transpose*/, 0 /*dst*/);
#pragma GCC unroll 4
    for (const auto row : math::fpu_row_offsets<FACE_R_DIM>())
    {
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + row, ADDR_MOD_0, FPU_MOV_ROWS, p_movb2a::SRCB_ROW32_OFFSET + row);
    }

    // The final pool adds into row 0, so clear it first.
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, EN_32BIT_DEST, 0 /*clear_zero_flags*/, ADDR_MOD_0, 0 /*where*/);

    mul_reduce_scalar_pool_(num_phases, ADDR_MOD_0);
}

/**
 * @brief Release the SrcA and SrcB banks held by the reduce tail.
 */
inline void _llk_math_mul_reduce_scalar_clear_dvalid_()
{
    using namespace ckernel;
    TTI_CLEARDVALID(
        p_cleardvalid::CLR_SRCAB_VLD, 0 /*cleardvalid_S*/, 0 /*dest_dvalid_reset*/, 0 /*dest_dvalid_client_bank_reset*/, 0 /*dest_pulse_last*/, 0 /*reset*/);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}
