// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_reduce_block.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

// Set by the last _llk_math_reduce_init_: the unpack thread was initialised with the same col_pair
inline bool reduce_col_pair_state = false;

// local function declarations
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape, const bool col_pair);

template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop(const ckernel::TensorShape& tensor_shape);

/**
 * @brief Transpose the pooled row result through SrcB so a row reduction lands as a column in the destination register.
 *
 * Only used for MAX pool (GMPOOL does column wise max of SrcA only).
 *
 * @tparam is_int_fpu_en: Cast int32 dest datums to int8 (via SFPU) before moving to SrcB.
 */
template <bool is_int_fpu_en>
inline void reduce_row_perform_transpose()
{
    // The MOVD2B/ELWADD below read the Src zero substitution flag (FlushDenormals = !flag): the caller sets it to preserve,
    // or a datum with a zero low byte would be flushed to 0 mid-reduction.
    if constexpr (is_int_fpu_en)
    {
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 0);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 2);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 2);
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
        TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x1);
    }

    // Move pooled result to SrcB rows 16–31 and transpose
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);

    if constexpr (is_int_fpu_en)
    {
        TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x0);
    }

    // Add transposed SrcB into Dest through a zeroed SrcA
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_ZEROSRC(0, 1, 0, 1);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
}

/**
 * @brief Pool one face into the destination register, dispatching to the right pool instruction for the op type.
 *
 * MAX uses GMPOOL; SUM/AVG use GAPOOL directly (LoFi) or the preconfigured MOP followed by a DVALID clear (high fidelity).
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam high_fidelity: Run the multi-phase fidelity MOP instead of a single GAPOOL.
 * @tparam clear_mode: Source-clear mode applied after pooling (p_setrwc::CLR_* value).
 * @tparam index: Destination-register offset (the GMPOOL/GAPOOL dst field) the pooled result is written to.
 */
template <PoolType type, bool high_fidelity, std::uint32_t clear_mode, std::uint32_t index = 0>
inline void reduce_pool_op()
{
    if constexpr (type == PoolType::MAX)
    {
        TTI_GMPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, index);
    }
    else if constexpr (high_fidelity)
    {
        ckernel_template::run();
        if constexpr (clear_mode != p_setrwc::CLR_NONE)
        {
            TTI_CLEARDVALID(clear_mode, 0);
        }
    }
    else
    {
        TTI_GAPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, index);
    }
}

/**
 * @brief Pool one row of faces, clearing SrcA/B between faces and keeping the final result.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam high_fidelity: Run the multi-phase fidelity MOP instead of a single GAPOOL.
 * @param num_faces_c_dim: Number of column faces in the row.
 */
template <PoolType type, bool high_fidelity>
inline void reduce_row_pool_all_faces(const std::uint32_t num_faces_c_dim)
{
    for (std::uint32_t col_num = 0; col_num < num_faces_c_dim - 1; col_num++)
    {
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 0>();
    }
    reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
}

template <PoolType type, bool high_fidelity>
inline void reduce_row_pool_all_faces_release(const std::uint32_t num_faces_c_dim)
{
    for (std::uint32_t col_num = 0; col_num < num_faces_c_dim; col_num++)
    {
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 0>();
    }
}

// The two faces of a face row sit in one SrcA bank (rows 0 and 16). Their pools go to different DEST rows, so the phases
// of the two chains alternate; per DEST row the phase order is the one of the face-by-face loop.
template <MathFidelity math_fidelity>
inline void reduce_col_pool_pair()
{
    constexpr std::uint32_t phases = (math_fidelity == MathFidelity::LoFi) ? 1 : to_underlying(math_fidelity);
    for (std::uint32_t phase = 0; phase < phases; phase++)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
        // The second face pools into DEST face 1 (dst field 16, the D counter value 8 of the face-by-face loop)
        if (phase + 1 < phases)
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 16);
        }
        else
        {
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_4, p_gpool::INDEX_DIS, 16);
        }
    }
}

/**
 * @brief Advance the dest counter to the next face row without releasing the source banks.
 *
 * @param is_narrow_tile: True when tile width < tile height (num_faces_c < num_faces_r), halving the stride.
 */
inline void reduce_row_advance_dest(const bool is_narrow_tile)
{
    if (!is_narrow_tile)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
}

// An immediate RMWCIB is one RISC instruction; the MMIO store of cfg_reg_rmw_tensix costs the math RISC more.
template <std::uint32_t value>
inline void reduce_row_max_set_src_zero_flag()
{
    static_assert(ALU_ACC_CTRL_Zero_Flag_disabled_src_SHAMT == 0 && ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK <= 0xff);
    TTI_RMWCIB0(ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK, value, ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32);
}

/**
 * @brief Pool the four faces of a 32x32 tile, face row 1 into DEST row 32 through the instruction's DEST offset; the last
 *        banks stay for the transposes unless `release`.
 */
template <bool release = false>
inline void reduce_row_max_pool_full_tile()
{
    // Each second pool waits for the first one's DEST row; a counter advance between the rows would add to that chain.
    TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
    TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
    TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 32);
    TTI_GMPOOL(release ? p_setrwc::CLR_AB : p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 32);
}

/**
 * @brief The moves of reduce_row_perform_transpose for both face rows of a 32x32 tile (DEST rows 0 and 32).
 */
inline void reduce_row_max_transpose_full_tile()
{
    // SrcB is addressed from B = 16: the counter writes issue while MOVD2B waits for the pools, not in its 3-cycle hold.
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 8, 0, p_setrwc::SET_BD);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_ZEROSRC(0, 1, 0, 1);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
    // B back to its carriage return (16), D to 0.
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 0, 0, p_setrwc::SET_BD);
    TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 32);
    TTI_TRNSPSRCB;
    TTI_MOVD2B(0, 0, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 32);
    TTI_ZEROSRC(0, 1, 0, 1);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 32);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 32);
}

/**
 * @brief REDUCE_SCALAR: transpose the partials of DEST row `row` into a SrcA column and pool it into DEST row 0, in the
 *        banks of the last pool (its SrcB row 0 is the scaler); releases them.
 */
template <PoolType type, MathFidelity math_fidelity, std::uint32_t row, bool clear_32b = false, bool clear_scratch = true>
inline void reduce_scalar_transpose_pool()
{
    // Need row in dest as column in src A
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);

    // copy over from dest to B and do transpose
    // use rows 16 - 31 in src B as scratch
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, row);
    TTI_TRNSPSRCB;
    // copy over all 16 rows from B to A
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    // zero out scratch in dest
    if constexpr (clear_scratch)
    {
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, clear_32b, 0, ADDR_MOD_0, row);
    }

    if constexpr (type == PoolType::MAX)
    {
        TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
    }
    else
    {
        if constexpr (is_high_fidelity(math_fidelity))
        {
            for (std::uint32_t i = 0; i < to_underlying(math_fidelity) - 1; i++)
            {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
            }
        }
        TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
    }
}

/**
 * @brief REDUCE_SCALAR: pool one face into DEST row `row`, the fidelity phases issued without the MOP.
 */
template <PoolType type, MathFidelity math_fidelity, std::uint32_t clear_mode, std::uint32_t row>
inline void reduce_scalar_pool_face()
{
    if constexpr (type == PoolType::MAX)
    {
        TTI_GMPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, row);
    }
    else
    {
        if constexpr (is_high_fidelity(math_fidelity))
        {
            for (std::uint32_t i = 0; i < to_underlying(math_fidelity) - 1; i++)
            {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, row);
            }
        }
        TTI_GAPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, row);
    }
}

/**
 * @brief REDUCE_SCALAR: pool faces [first, num_faces) into DEST row `row`; the last one keeps its banks when keep_last.
 */
template <PoolType type, MathFidelity math_fidelity, std::uint32_t row>
inline void reduce_scalar_pool_faces(const std::uint32_t first, const std::uint32_t num_faces, const bool keep_last)
{
    if (first >= num_faces)
    {
        return;
    }
    for (std::uint32_t face = first; face < num_faces - 1; face++)
    {
        reduce_scalar_pool_face<type, math_fidelity, p_setrwc::CLR_AB, row>();
    }
    if (keep_last)
    {
        reduce_scalar_pool_face<type, math_fidelity, p_setrwc::CLR_NONE, row>();
    }
    else
    {
        reduce_scalar_pool_face<type, math_fidelity, p_setrwc::CLR_AB, row>();
    }
}

/**
 * @brief REDUCE_SCALAR: transpose the partials of DEST row `row` into SrcA rows [srca_row, srca_row + 16), so the rows
 *        below keep the bank's faces.
 */
template <std::uint32_t row, bool clear_32b, std::uint32_t srca_row = 16>
inline void reduce_scalar_transpose_upper()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, row);
    TTI_TRNSPSRCB;
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + srca_row + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + srca_row + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + srca_row + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + srca_row + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, clear_32b, 0, ADDR_MOD_0, row);
}

/**
 * @brief REDUCE_SCALAR SUM/AVG: one face into DEST row `row` and the previous tile's final pool into row 0, phase by
 *        phase: two chains on different rows, each in its own order. Releases the banks unless keep.
 */
template <MathFidelity math_fidelity, std::uint32_t row>
inline void reduce_scalar_face_with_final_pool(const bool keep)
{
    for (std::uint32_t i = 0; i < (is_high_fidelity(math_fidelity) ? to_underlying(math_fidelity) : 1u) - 1; i++)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, row);
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
    }
    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, row);
    if (keep)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_4, p_gpool::INDEX_DIS, 0);
    }
    else
    {
        TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_4, p_gpool::INDEX_DIS, 0);
    }
}

/**
 * @brief REDUCE_SCALAR SUM/AVG, a block's tile after the first: the previous tile (`prev_row`) is transposed and pooled
 *        in face 1's banks alongside face 1.
 */
template <PoolType type, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, std::uint32_t row, std::uint32_t prev_row>
inline void reduce_scalar_block_tile(const std::uint32_t num_faces, const bool last)
{
    reduce_scalar_pool_face<type, math_fidelity, p_setrwc::CLR_AB, row>();
    reduce_scalar_transpose_upper<prev_row, is_fp32_dest_acc_en>();
    reduce_scalar_face_with_final_pool<math_fidelity, row>(last && num_faces == 2);
    reduce_scalar_pool_faces<type, math_fidelity, row>(2, num_faces, last);
}

// REDUCE_SCALAR SUM/AVG block in tile pairs: the unpacker puts one face of tiles a and b in one SrcA bank (rows 0-15, 16-31), so their
// chains (DEST rows ra, rb) issue alternately; the previous pair's transposed partials wait in rows 32-47 for the final pools.
template <MathFidelity math_fidelity>
constexpr std::uint32_t reduce_pair_phases = is_high_fidelity(math_fidelity) ? to_underlying(math_fidelity) : 1;

// A final pool lands in row 0 when the block accumulates into one tile; with independent outputs (indep) in the output row fdst.
template <bool indep, std::uint32_t clear_mode, std::uint32_t addr_mod>
inline void reduce_pair_final_pool(const std::uint32_t fdst)
{
    if constexpr (indep)
    {
        TT_GAPOOL(clear_mode, p_gpool::DIM_16X16, addr_mod, p_gpool::INDEX_DIS, fdst);
    }
    else
    {
        TTI_GAPOOL(clear_mode, p_gpool::DIM_16X16, addr_mod, p_gpool::INDEX_DIS, 0);
    }
}

// One bank of a pair; final_pool also pools the previous tile's transposed partials (SrcA row 32) into row 0.
template <MathFidelity math_fidelity, std::uint32_t ra, std::uint32_t rb, bool final_pool, bool indep = false>
inline void reduce_pair_bank(const bool release, const std::uint32_t fdst = 0)
{
    constexpr std::uint32_t phases = reduce_pair_phases<math_fidelity>;
    for (std::uint32_t phase = 0; phase + 1 < phases; phase++)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, ra);
        if constexpr (final_pool)
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, rb);
            reduce_pair_final_pool<indep, p_setrwc::CLR_NONE, ADDR_MOD_3>(fdst);
        }
        else
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, rb);
        }
    }
    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, ra);
    if constexpr (final_pool)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, rb);
    }
    if constexpr (final_pool)
    {
        if (release)
        {
            reduce_pair_final_pool<indep, p_setrwc::CLR_AB, ADDR_MOD_4>(fdst);
        }
        else
        {
            reduce_pair_final_pool<indep, p_setrwc::CLR_NONE, ADDR_MOD_4>(fdst);
        }
    }
    else if (release)
    {
        TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_4, p_gpool::INDEX_DIS, rb);
    }
    else
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_4, p_gpool::INDEX_DIS, rb);
    }
}

// The same for an odd last tile: chain a only.
template <MathFidelity math_fidelity, std::uint32_t ra, bool final_pool, bool indep = false>
inline void reduce_pair_bank_a(const bool release, const std::uint32_t fdst = 0)
{
    constexpr std::uint32_t phases = reduce_pair_phases<math_fidelity>;
    for (std::uint32_t phase = 0; phase + 1 < phases; phase++)
    {
        if constexpr (final_pool)
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_5, p_gpool::INDEX_DIS, ra);
            reduce_pair_final_pool<indep, p_setrwc::CLR_NONE, ADDR_MOD_3>(fdst);
        }
        else
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, ra);
        }
    }
    if constexpr (final_pool)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_5, p_gpool::INDEX_DIS, ra);
    }
    if constexpr (final_pool)
    {
        if (release)
        {
            reduce_pair_final_pool<indep, p_setrwc::CLR_AB, ADDR_MOD_4>(fdst);
        }
        else
        {
            reduce_pair_final_pool<indep, p_setrwc::CLR_NONE, ADDR_MOD_4>(fdst);
        }
    }
    else if (release)
    {
        TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, ra);
    }
    else
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, ra);
    }
}

template <MathFidelity math_fidelity, std::uint32_t ra, std::uint32_t rb, bool final_pool, bool indep = false>
inline void reduce_pair_bank_of(const bool has_b, const bool release, const std::uint32_t fdst = 0)
{
    if (has_b)
    {
        reduce_pair_bank<math_fidelity, ra, rb, final_pool, indep>(release, fdst);
    }
    else
    {
        reduce_pair_bank_a<math_fidelity, ra, final_pool, indep>(release, fdst);
    }
}

// The last pair's final pools, its partials transposed over the data of the kept bank (a in rows 0-15, b in rows 16-31).
template <MathFidelity math_fidelity, bool clear_32b, std::uint32_t ra, std::uint32_t rb, bool indep = false>
inline void reduce_pair_tail(const bool has_b, const std::uint32_t fa = 0, const std::uint32_t fb = 0)
{
    constexpr std::uint32_t phases = reduce_pair_phases<math_fidelity>;
    reduce_scalar_transpose_upper<ra, clear_32b, 0>();
    if (has_b)
    {
        reduce_scalar_transpose_upper<rb, clear_32b, 16>();
    }
    for (std::uint32_t phase = 0; phase + 1 < phases; phase++)
    {
        reduce_pair_final_pool<indep, p_setrwc::CLR_NONE, ADDR_MOD_1>(fa);
    }
    if (has_b)
    {
        reduce_pair_final_pool<indep, p_setrwc::CLR_NONE, ADDR_MOD_6>(fa);
        for (std::uint32_t phase = 0; phase + 1 < phases; phase++)
        {
            reduce_pair_final_pool<indep, p_setrwc::CLR_NONE, ADDR_MOD_1>(fb);
        }
        reduce_pair_final_pool<indep, p_setrwc::CLR_AB, ADDR_MOD_4>(fb);
    }
    else
    {
        reduce_pair_final_pool<indep, p_setrwc::CLR_AB, ADDR_MOD_4>(fa);
    }
}

// One pair on DEST rows (ra, rb); (pa, pb) are the previous pair's.
template <MathFidelity math_fidelity, bool clear_32b, std::uint32_t ra, std::uint32_t rb, std::uint32_t pa, std::uint32_t pb, bool indep = false>
inline void reduce_pair(const bool first, const bool has_b, const bool last, const std::uint32_t fa = 0, const std::uint32_t fb = 0)
{
    reduce_pair_bank_of<math_fidelity, ra, rb, false>(has_b, true);
    if (first)
    {
        reduce_pair_bank_of<math_fidelity, ra, rb, false>(has_b, true);
        reduce_pair_bank_of<math_fidelity, ra, rb, false>(has_b, true);
    }
    else
    {
        reduce_scalar_transpose_upper<pa, clear_32b, 32>();
        reduce_pair_bank_of<math_fidelity, ra, rb, true, indep>(has_b, true, fa);
        reduce_scalar_transpose_upper<pb, clear_32b, 32>();
        reduce_pair_bank_of<math_fidelity, ra, rb, true, indep>(has_b, true, fb);
    }
    reduce_pair_bank_of<math_fidelity, ra, rb, false>(has_b, !last);
}

// With dst_stride != 0 every tile is its own output, in DEST tile (dst_index + tile * dst_stride); the final pools write there.
template <PoolType type, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool indep = false>
inline void reduce_scalar_pair_block(const std::uint32_t dst_index, const std::uint32_t num_tiles, const std::uint32_t dst_stride = 0)
{
    const std::uint32_t row_stride = dst_stride << 6;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // SrcA steps: +16 (next tile), row 0 with the next phase, row 0 with phase 0, +32 (to the partials), +16 with phase 0
    addr_mod_t {
        .srca = {.incr = 16},
    }
        .set(ADDR_MOD_2);
    addr_mod_t {
        .srca     = {.clr = 1},
        .fidelity = {.incr = 1},
    }
        .set(ADDR_MOD_3);
    addr_mod_t {
        .srca     = {.clr = 1},
        .fidelity = {.clr = 1},
    }
        .set(ADDR_MOD_4);
    addr_mod_t {
        .srca = {.incr = 32},
    }
        .set(ADDR_MOD_5);
    addr_mod_t {
        .srca     = {.incr = 16},
        .fidelity = {.clr = 1},
    }
        .set(ADDR_MOD_6);

    // Pairs alternate their DEST rows, so a pair's chains start while the previous pair's partials are being transposed
    const std::uint32_t pairs = (num_tiles + 1) / 2;
    for (std::uint32_t pair = 0; pair < pairs; pair++)
    {
        const bool first       = pair == 0;
        const bool last        = pair == pairs - 1;
        const bool has_b       = !(last && (num_tiles & 1));
        const std::uint32_t fa = (2 * pair - 2) * row_stride;
        if (pair & 1)
        {
            reduce_pair<math_fidelity, is_fp32_dest_acc_en, 24, 32, 8, 16, indep>(first, has_b, last, fa, fa + row_stride);
        }
        else
        {
            reduce_pair<math_fidelity, is_fp32_dest_acc_en, 8, 16, 24, 32, indep>(first, has_b, last, fa, fa + row_stride);
        }
    }
    const std::uint32_t last_a = 2 * (pairs - 1) * row_stride;
    if ((pairs - 1) & 1)
    {
        reduce_pair_tail<math_fidelity, is_fp32_dest_acc_en, 24, 32, indep>(!(num_tiles & 1), last_a, last_a + row_stride);
    }
    else
    {
        reduce_pair_tail<math_fidelity, is_fp32_dest_acc_en, 8, 16, indep>(!(num_tiles & 1), last_a, last_a + row_stride);
    }
}

/**
 * @brief REDUCE_SCALAR block accumulated into one DEST tile, faces >= 2. MAX transposes once per block (exact in any
 *        order). SUM/AVG keeps the tile path's order of additions, but tile t's transpose runs in tile t+1's banks, so the
 *        unpacker fills the free banks meanwhile; its final pool reads face 1's scaler row instead of face 3's, the
 *        same values for a scaler whose faces share their first row.
 */
template <PoolType type, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void reduce_scalar_block_accumulate(const std::uint32_t dst_index, const std::uint32_t num_tiles, const std::uint32_t num_faces)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (type == PoolType::MAX)
    {
        // GMPOOL takes the max with the scratch row, so a tile copied into DST must leave it 0, not minus infinity.
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 4);
        for (std::uint32_t tile = 0; tile < num_tiles; tile++)
        {
            reduce_scalar_pool_faces<type, math_fidelity, 4>(0, num_faces, tile == num_tiles - 1);
        }
        reduce_scalar_transpose_pool<type, math_fidelity, 4>();
    }
    else
    {
        // SrcA to the transposed rows (16), back to the face (0) with the next phase, back to 0 and phase 0; the tile path
        // does not use these three.
        addr_mod_t {
            .srca = {.incr = 16},
        }
            .set(ADDR_MOD_2);
        addr_mod_t {
            .srca     = {.clr = 1},
            .fidelity = {.incr = 1},
        }
            .set(ADDR_MOD_3);
        addr_mod_t {
            .srca     = {.clr = 1},
            .fidelity = {.clr = 1},
        }
            .set(ADDR_MOD_4);

        // Scratch rows in other 8-row DEST blocks than row 0, so the alternating pools do not wait on each other; with a
        // 32-bit DEST a 16-bit mode ZEROACC would clear another row than these.
        reduce_scalar_pool_faces<type, math_fidelity, 8>(0, num_faces, false);
        for (std::uint32_t tile = 1; tile < num_tiles; tile++)
        {
            const bool last = tile == num_tiles - 1;
            if (tile & 1)
            {
                reduce_scalar_block_tile<type, is_fp32_dest_acc_en, math_fidelity, 16, 8>(num_faces, last);
            }
            else
            {
                reduce_scalar_block_tile<type, is_fp32_dest_acc_en, math_fidelity, 8, 16>(num_faces, last);
            }
        }
        if ((num_tiles - 1) & 1)
        {
            reduce_scalar_transpose_pool<type, math_fidelity, 16, is_fp32_dest_acc_en>();
        }
        else
        {
            reduce_scalar_transpose_pool<type, math_fidelity, 8, is_fp32_dest_acc_en>();
        }
    }
}

/**
 * @brief Configure the reduce MOP for all paths: REDUCE_ROW SUM/AVG records a dual replay-buffer
 *        MOP, high-fidelity COL/SCALAR programs a multi-phase GAPOOL loop, all other paths need no MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape determining face count and dest stride.
 */
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop(const ckernel::TensorShape& tensor_shape)
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        if (tensor_shape.total_num_faces() == 4)
        {
            constexpr std::uint32_t replay_buf_len = 8;
            constexpr std::uint32_t last_clr       = high_fidelity ? p_setrwc::CLR_NONE : p_setrwc::CLR_AB;
            constexpr std::uint32_t inner_loops    = high_fidelity ? to_underlying(math_fidelity) : 1;
            const std::uint32_t replay_start       = ckernel::math::replay_buf_offset;

            load_replay_buf(
                replay_start,
                replay_buf_len,
                []
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                    TTI_MVMUL(last_clr, 0, ADDR_MOD_3, 0);
                });

            ckernel_template tmp(1, inner_loops, lltt::replay_insn(replay_start, replay_buf_len));
            if constexpr (high_fidelity)
            {
                tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
            }
            tmp.program();
        }
        else
        {
            constexpr std::uint32_t fid_count  = (math_fidelity == MathFidelity::LoFi) ? 1 : to_underlying(math_fidelity);
            const std::uint32_t replay_buf_len = tensor_shape.num_faces_c_dim * 2 * fid_count;
            const std::uint32_t replay_start   = ckernel::math::replay_buf_offset;

            load_replay_buf(
                replay_start,
                replay_buf_len,
                [&tensor_shape]
                {
                    for (std::uint32_t faces_remaining = tensor_shape.num_faces_c_dim; faces_remaining > 0; faces_remaining--)
                    {
                        for (std::uint32_t p = 0; p < fid_count - 1; p++)
                        {
                            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                        }
                        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);

                        for (std::uint32_t p = 0; p < fid_count - 1; p++)
                        {
                            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 8);
                        }
                        if (faces_remaining == 1)
                        {
                            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 8);
                        }
                        else
                        {
                            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_2, 8);
                        }
                    }
                });

            const std::uint32_t replay_op = lltt::replay_insn(replay_start, replay_buf_len);
            ckernel_template tmp(tensor_shape.num_faces_r_dim, 1, replay_op);
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_B));
            tmp.program();
        }
    }
    else if constexpr (high_fidelity)
    {
        constexpr int inner_loop_len = to_underlying(math_fidelity);
        constexpr std::uint32_t dst  = (dim == ReduceDim::REDUCE_SCALAR) ? 4 : 0;

        ckernel_template tmp(1, inner_loop_len, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, dst));
        tmp.set_last_inner_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, dst));
        tmp.set_last_outer_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, dst));
        tmp.program();
    }
}

/**
 * @brief Perform a reduction on the math thread, pooling faces into the destination register.
 *
 * REDUCE_ROW SUM/AVG uses MVMUL to multiply each face row by a column-scaler in SrcA, producing the reduced
 * column directly in dest; MAX pools each face row then transposes the result into dest via SrcB.
 * REDUCE_COL pools down rows of faces; REDUCE_SCALAR pools all faces then transposes the partial result
 * into a single column for a final pool.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam is_int_fpu_en: Enable integer FPU datapath (casts int32 dest datums to int8 before moving to SrcB).
 * @param dst_index: Tile index into the destination register.
 * @param tensor_shape: Tensor shape describing tile dimensions.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this
 *       function, and @ref _llk_math_reduce_uninit_ after it to restore modified state.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool is_int_fpu_en = false, int col_pair_static = -1>
inline void _llk_math_reduce_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape)
{
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_reduce_", tensor_shape);

    // Supported narrow tiles per BH Tiny Tile Summary: [16]x16 (num_faces=1) and [32]x16 (num_faces=2) only
    LLK_ASSERT(
        !((tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */ && tensor_shape.total_num_faces() == 4),
        "Reduce narrow tile requires num_faces 1 or 2; num_faces=4 is full-width 32x32");

    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        const bool is_narrow_tile = tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim;

        if constexpr (type == PoolType::MAX)
        {
            // The pools run under the operand default of the Src zero flag and the transposes under preserve, so both face
            // rows are pooled first; the last face's banks stay for the transposes.
            // Constant config writes: the tracked setters' calls and checks starve the math thread's instruction issue.
            const bool default_is_preserve = requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt);
            const bool full_tile           = !is_int_fpu_en && (tensor_shape.total_num_faces() == 4);
            if (default_is_preserve)
            {
                math::_configure_preserve_zero_flag_state_();
            }
            else
            {
                reduce_row_max_set_src_zero_flag<0>();
            }
            if (full_tile)
            {
                reduce_row_max_pool_full_tile();
            }
            else
            {
                if (tensor_shape.num_faces_r_dim > 1)
                {
                    reduce_row_pool_all_faces_release<type, high_fidelity>(tensor_shape.num_faces_c_dim);
                    reduce_row_advance_dest(is_narrow_tile);
                }
                reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
            }

            if (!default_is_preserve)
            {
                reduce_row_max_set_src_zero_flag<1>();
            }
            math::src_zero_flag_hw = 1;
            if (full_tile)
            {
                reduce_row_max_transpose_full_tile();
            }
            else
            {
                if (tensor_shape.num_faces_r_dim > 1)
                {
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                    reduce_row_perform_transpose<is_int_fpu_en>();
                    reduce_row_advance_dest(is_narrow_tile);
                }
                reduce_row_perform_transpose<is_int_fpu_en>();
            }
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
        }
        else
        {
            ckernel_template::run();
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        // col_pair_static 1: the caller's init opted in, so the state need not be read
        if constexpr (type != PoolType::MAX && math_fidelity != MathFidelity::LoFi && col_pair_static != 0)
        {
            if ((col_pair_static == 1 || reduce_col_pair_state) && reduce_col_paired<type, dim>(tensor_shape))
            {
                for (std::uint32_t row_num = 0; row_num < tensor_shape.num_faces_r_dim; row_num++)
                {
                    reduce_col_pool_pair<math_fidelity>();
                }
                return;
            }
        }
        for (std::uint32_t row_num = 0; row_num < tensor_shape.num_faces_r_dim; row_num++)
        {
            // Just pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            if (tensor_shape.num_faces_c_dim > 1)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
                TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

                reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            }
            // Reset Dest Counter
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AD);
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        if constexpr (type == PoolType::MAX)
        {
            // GMPOOL takes the max with the scratch row, so a tile copied into DST must leave it 0, not minus infinity.
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 4);
        }
        for (std::uint32_t face_num = 0; face_num < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face_num++)
        {
            // Wait and pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 4>();
        }
        // Wait and pool
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 4>();

        // A max clears its scratch row on entry, so the clear before the final pool would only repeat it.
        reduce_scalar_transpose_pool<type, math_fidelity, 4, false, type != PoolType::MAX>();
    }
}

// REDUCE_ROW MAX of full tiles accumulating into one DEST tile: every tile is pooled into DEST rows 0 and 32, and the
// partials are transposed once (a max is exact in any order).
inline void reduce_row_max_block_full_tile(const std::uint32_t dst_index, const std::uint32_t num_tiles)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    const bool default_is_preserve = requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt);
    if (default_is_preserve)
    {
        math::_configure_preserve_zero_flag_state_();
    }
    else
    {
        reduce_row_max_set_src_zero_flag<0>();
    }
    for (std::uint32_t tile = 0; tile + 1 < num_tiles; tile++)
    {
        reduce_row_max_pool_full_tile<true>();
    }
    reduce_row_max_pool_full_tile();
    if (!default_is_preserve)
    {
        reduce_row_max_set_src_zero_flag<1>();
    }
    math::src_zero_flag_hw = 1;
    reduce_row_max_transpose_full_tile();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
}

// A dst_stride of 0 accumulates the block into one tile; scalar_pair is the decision for the unpacker's pair layout (SCALAR SUM/AVG,
// full tiles).
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool is_int_fpu_en = false>
inline void _llk_math_reduce_block_(
    const std::uint32_t dst_index,
    std::uint32_t num_tiles,
    const std::uint32_t dst_stride,
    const ckernel::TensorShape tensor_shape,
    const bool scalar_pair = false)
{
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_reduce_block_", tensor_shape);

    if constexpr (dim == ReduceDim::REDUCE_ROW && type == PoolType::MAX)
    {
        if (dst_stride == 0 && num_tiles > 1 && !is_int_fpu_en && tensor_shape.total_num_faces() == 4)
        {
            reduce_row_max_block_full_tile(dst_index, num_tiles);
            return;
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        if (num_tiles > 1 && reduce_block_holds_scaler<type, dim>(tensor_shape))
        {
            constexpr bool high_fidelity     = is_high_fidelity(math_fidelity);
            const std::uint32_t replay_start = ckernel::math::replay_buf_offset;
            std::uint32_t dst                = dst_index;

            while (num_tiles > 0)
            {
                const std::uint32_t chunk = num_tiles < REDUCE_BLOCK_MAX_TILES ? num_tiles : REDUCE_BLOCK_MAX_TILES;
                for (std::uint32_t tile = 0; tile < chunk; tile++)
                {
                    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst);
                    // The init's MVMUL record, with the source release narrowed to the data bank (SrcB)
                    if constexpr (high_fidelity)
                    {
                        for (std::uint32_t phase = 0; phase < to_underlying(math_fidelity); phase++)
                        {
                            lltt::replay(replay_start, 8);
                        }
                        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                    }
                    else
                    {
                        lltt::replay(replay_start, 7);
                        TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_3, 0);
                    }
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
                    dst += dst_stride;
                }
                TTI_CLEARDVALID(p_setrwc::CLR_A, 0);
                num_tiles -= chunk;
            }
            return;
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        if constexpr (type != PoolType::MAX)
        {
            if (dst_stride != 0 && num_tiles > 1 && scalar_pair && reduce_scalar_paired<type, dim>(tensor_shape))
            {
                reduce_scalar_pair_block<type, is_fp32_dest_acc_en, math_fidelity, true>(dst_index, num_tiles, dst_stride);
                return;
            }
        }
        if (dst_stride == 0 && num_tiles > 1 && tensor_shape.total_num_faces() > 1)
        {
            if constexpr (type != PoolType::MAX)
            {
                if (scalar_pair && reduce_scalar_paired<type, dim>(tensor_shape))
                {
                    reduce_scalar_pair_block<type, is_fp32_dest_acc_en, math_fidelity>(dst_index, num_tiles);
                    return;
                }
            }
            reduce_scalar_block_accumulate<type, is_fp32_dest_acc_en, math_fidelity>(dst_index, num_tiles, tensor_shape.total_num_faces());
            return;
        }
    }

    for (std::uint32_t tile = 0; tile < num_tiles; tile++)
    {
        _llk_math_reduce_<type, dim, is_fp32_dest_acc_en, math_fidelity, is_int_fpu_en>(dst_index + tile * dst_stride, tensor_shape);
    }
}

/**
 * @brief Program address mod registers for the reduce operation.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape determining dest stride for REDUCE_ROW.
 */
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape, const bool col_pair)
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        if (tensor_shape.total_num_faces() == 4)
        {
            addr_mod_t {
                .srcb = {.incr = 8},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_0);

            addr_mod_t {
                .srcb = {.incr = 24},
                .dest = {.incr = 24},
            }
                .set(ADDR_MOD_1);

            addr_mod_t {
                .srca = {.incr = 16},
                .srcb = {.incr = 16, .cr = 1},
                .dest = {.cr = 1},
            }
                .set(ADDR_MOD_2);

            addr_mod_t {
                .srca     = {.clr = 1, .cr = 1},
                .srcb     = {.clr = 1, .cr = 1},
                .dest     = {.clr = 1, .cr = 1},
                .fidelity = {.incr = high_fidelity ? 1u : 0u},
            }
                .set(ADDR_MOD_3);
        }
        else
        {
            if constexpr (high_fidelity)
            {
                addr_mod_t {
                    .fidelity = {.incr = 1},
                }
                    .set(ADDR_MOD_0);
            }

            addr_mod_t {
                .srcb     = {.incr = 8},
                .fidelity = {.clr = 1},
            }
                .set(ADDR_MOD_1);

            addr_mod_t {
                .srcb     = {.cr = 1},
                .fidelity = {.clr = 1},
            }
                .set(ADDR_MOD_2);

            if (tensor_shape.num_faces_c_dim == 2)
            {
                addr_mod_t {
                    .srcb     = {.cr = 1},
                    .dest     = {.incr = 32},
                    .fidelity = {.clr = 1},
                }
                    .set(ADDR_MOD_3);
            }
            else
            {
                addr_mod_t {
                    .srcb     = {.cr = 1},
                    .dest     = {.incr = 16},
                    .fidelity = {.clr = 1},
                }
                    .set(ADDR_MOD_3);
            }
        }
    }
    else
    {
        addr_mod_t {
            .fidelity = {.clr = 1},
        }
            .set(ADDR_MOD_0);

        if constexpr (type == PoolType::MAX)
        {
            addr_mod_t {
                .srcb = {.incr = 8},
                .dest = {.incr = 8},
            }
                .set(ADDR_MOD_1);
        }
        else if constexpr (high_fidelity)
        {
            addr_mod_t {
                .fidelity = {.incr = 1},
            }
                .set(ADDR_MOD_1);
        }

        if constexpr (dim == ReduceDim::REDUCE_COL && type != PoolType::MAX)
        {
            if (col_pair)
            {
                addr_mod_t {
                    .srca = {.incr = FACE_R_DIM},
                }
                    .set(ADDR_MOD_2);

                addr_mod_t {
                    .srca     = {.clr = 1},
                    .fidelity = {.incr = 1},
                }
                    .set(ADDR_MOD_3);

                addr_mod_t {
                    .srca     = {.clr = 1},
                    .fidelity = {.clr = 1},
                }
                    .set(ADDR_MOD_4);
            }
        }
    }
}

/**
 * @brief Configure the math (FPU) thread for a reduce operation: programs address mods and the MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape describing tile dimensions.
 * @note Call @ref _llk_math_reduce_ with matching template args after this function,
 *       and @ref _llk_math_reduce_uninit_ after the last reduce to restore modified state.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool col_pair = false>
inline void _llk_math_reduce_init_(const ckernel::TensorShape& tensor_shape)
{
    // There is no min-pool instruction - the FPU has GMPOOL (max) and GAPOOL (average/sum).
    static_assert(
        type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    static_assert(!col_pair || math_fidelity != MathFidelity::LoFi, "the column pair layout is not used at LoFi");
    bool pair = false;
    if constexpr (dim == ReduceDim::REDUCE_COL && type != PoolType::MAX && math_fidelity != MathFidelity::LoFi)
    {
        pair                  = col_pair && reduce_col_paired<type, dim>(tensor_shape);
        reduce_col_pair_state = pair;
    }
    reduce_configure_addrmod<type, dim, math_fidelity>(tensor_shape, pair);
    reduce_configure_mop<type, dim, math_fidelity>(tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);

    // Establish the operand-driven DEFAULT zero-flag state before the reduce's GMPOOLs, mirroring
    // _llk_math_matmul_init_ / _llk_math_eltwise_binary_init_. A preceding copy_init that left
    // PRESERVE (keep denormals) would otherwise leak "keep" into the pool GMPOOL — harmless on HW
    // when fp32 DEST accumulation is enabled (the flag is ignored), but a real invariant violation.
    math::_configure_default_zero_flag_state_();
}

/**
 * @brief Uninitialize after a reduce operation, undoing any init/execute-time workarounds.
 *
 * @note Reverses @ref _llk_math_reduce_init_
 */
template <PoolType type = PoolType::MAX, ReduceDim dim = ReduceDim::REDUCE_ROW>
inline void _llk_math_reduce_uninit_()
{
    // Only the ROW MAX pools leave the zero flag changed
    if constexpr (type == PoolType::MAX && dim == ReduceDim::REDUCE_ROW)
    {
        math::_configure_default_zero_flag_state_();
    }
}
