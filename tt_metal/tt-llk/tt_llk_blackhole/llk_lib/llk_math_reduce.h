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
#include "tensor_shape.h"

using namespace ckernel;

// local function declarations
template <PoolType type, MathFidelity math_fidelity>
inline void reduce_configure_addrmod();

template <ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop();

/**
 * @brief Transpose the pooled row result through SrcB so a row-reduction lands as a column in the destination register.
 *
 * Only used for MAX pool (GMPOOL does column-wise max of SrcA only, cannot use the operand-swap path).
 *
 * @tparam is_int_fpu_en: Cast int32 dest datums to int8 (via SFPU) before moving to SrcB.
 */
template <bool is_int_fpu_en>
inline void reduce_row_perform_transpose()
{
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
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
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
    // Pool face from SrcA (REDUCE_ROW: scaler; COL/SCALAR: data)
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
 * @brief Pool one face for swapped REDUCE_ROW (data in SrcB), iterating over all 16 SrcB rows.
 *
 * GAPOOL processes one SrcB row per invocation. With swapped operands (data→SrcB, scaler→SrcA),
 * we must call GAPOOL once per SrcB row (×fidelity phases for HiFi) and advance SrcB/Dest counters
 * between rows via ADDR_MOD_1 (srcb+1, dest+1, fidelity clear).
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam clear_mode: Source-clear mode applied after pooling the last row.
 */
template <MathFidelity math_fidelity, std::uint32_t clear_mode>
inline void reduce_row_swapped_pool_face()
{
    constexpr bool high_fidelity     = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t num_rows = FACE_R_DIM;

    for (std::uint32_t row = 0; row < num_rows - 1; row++)
    {
        if constexpr (high_fidelity)
        {
            for (std::uint32_t f = 0; f < to_underlying(math_fidelity) - 1; f++)
            {
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
            }
        }
        // Last fidelity phase for this row: advance SrcB+1, Dest+1, clear fidelity
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
    }
    // Last row: run fidelity phases, then apply the caller's clear_mode
    if constexpr (high_fidelity)
    {
        for (std::uint32_t f = 0; f < to_underlying(math_fidelity) - 1; f++)
        {
            TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
        }
    }
    TTI_GAPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
}

/**
 * @brief Pool one row of faces, clearing SrcA/B between faces and keeping the final result.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam high_fidelity: Run the multi-phase fidelity MOP instead of a single GAPOOL.
 * @param num_faces_c_dim: Number of column-faces in the row.
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

/**
 * @brief Pool one row of faces for swapped REDUCE_ROW, clearing SrcA/B and resetting Dest between faces.
 *
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param num_faces_c_dim: Number of column-faces in the row.
 */
template <MathFidelity math_fidelity>
inline void reduce_row_swapped_pool_all_faces(const std::uint32_t num_faces_c_dim)
{
    for (std::uint32_t col_num = 0; col_num < num_faces_c_dim - 1; col_num++)
    {
        reduce_row_swapped_pool_face<math_fidelity, p_setrwc::CLR_AB>();
        // SrcB=15, Dest=15 after 16 rows. Reset both to 0 so next face accumulates into same Dest positions.
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);
    }
    reduce_row_swapped_pool_face<math_fidelity, p_setrwc::CLR_AB>();
    // Reset SrcB and Dest to 0 after last face too (Dest must be at 0 for advance_dest arithmetic)
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);
}

/**
 * @brief Advance the dest write pointer to the next face row and clear SrcA/B.
 *
 * @param is_narrow_tile: True when the tile has fewer column-faces than row-faces.
 */
inline void reduce_row_advance_dest(const bool is_narrow_tile)
{
    if (!is_narrow_tile)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_BD);
}

/**
 * @brief Perform a reduction on the math thread, pooling faces into the destination register.
 *
 * For REDUCE_ROW with SUM/AVG, operands are swapped at unpack (scaler→SrcA, data→SrcB) so GAPOOL produces
 * row sums directly in the correct column positions. For MAX, the original layout is used with a post-pool
 * transpose (GMPOOL only reads SrcA). REDUCE_COL pools down rows of faces;
 * REDUCE_SCALAR pools all faces then transposes the partial result into a single column for a final pool.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam is_int_fpu_en: Enable integer FPU datapath (casts int32 dest datums to int8 before moving to SrcB).
 * @tparam enforce_fp32_accumulation: Force FP32 accumulation through the transpose (requires is_fp32_dest_acc_en).
 * @param dst_index: Tile index into the destination register.
 * @param tensor_shape: Tensor shape describing tile dimensions.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this
 *       function, and @ref _llk_math_reduce_uninit_ after it to restore modified state.
 */
template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool is_int_fpu_en             = false,
    bool enforce_fp32_accumulation = false>
inline void _llk_math_reduce_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Supported narrow tiles per BH Tiny Tile Summary: [16]x16 (num_faces=1) and [32]x16 (num_faces=2) only
    LLK_ASSERT(
        !((tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */ && tensor_shape.total_num_faces() == 4),
        "Reduce narrow tile requires num_faces 1 or 2; num_faces=4 is full-width 32x32");

    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        // Stall math until SrcA/SrcB banks are available and packer is idle
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD | p_stall::PACK);

        const bool is_narrow_tile = tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim;

        if constexpr (type == PoolType::MAX)
        {
            // GMPOOL does column-wise max of SrcA only — cannot swap operands.
            // Data is transposed at unpack (haloize), pool produces 1×16 row, transpose back to column.
            reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
            reduce_row_perform_transpose<is_int_fpu_en>();

            if (tensor_shape.num_faces_r_dim > 1)
            {
                reduce_row_advance_dest(is_narrow_tile);
                reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
                reduce_row_perform_transpose<is_int_fpu_en>();
            }
        }
        else
        {
            reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);

            if (tensor_shape.num_faces_r_dim > 1)
            {
                reduce_row_advance_dest(is_narrow_tile);
                reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
            }
        }

        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
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
        for (std::uint32_t face_num = 0; face_num < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face_num++)
        {
            // Wait and pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 4>();
        }
        // Wait and pool
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 4>();

        // Need row in dest as column in src A
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);

        // copy over from dest to B and do transpose
        // use rows 16 - 31 in src B as scratch
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 4);
        TTI_GATESRCRST(0b1, 0b1);
        TTI_TRNSPSRCB;
        // gate math instructions until src B has been updated
        TTI_GATESRCRST(0b1, 0b1);
        // copy over all 16 rows from B to A
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
        // gate math instructions until src A has been updated by MOV instructions
        TTI_GATESRCRST(0b1, 0b1);
        // zero out scratch in dest
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 4);

        if constexpr (type == PoolType::MAX)
        {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
        else
        {
            if constexpr (high_fidelity)
            {
                for (std::uint32_t i = 0; i < to_underlying(math_fidelity) - 1; i++)
                {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
                }
            }
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
    }
}

/**
 * @brief Program the address-mod slots for a reduce: no-op/fidelity-clear, single-row, 8-row, and (high fidelity) fidelity-step.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <PoolType type, MathFidelity math_fidelity>
inline void reduce_configure_addrmod()
{
    constexpr bool high_fidelity               = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);

    addr_mod_t {
        .srca     = {.incr = 0},
        .srcb     = {.incr = 1},
        .dest     = {.incr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);

    if constexpr (high_fidelity)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity_increment}}.set(ADDR_MOD_3);
    }
}

/**
 * @brief Build the high-fidelity reduce MOP: a multi-phase GAPOOL sequence (one inner-loop iteration per fidelity phase).
 *
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision; sets the inner-loop length, values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop()
{
    constexpr int inner_loop_len = to_underlying(math_fidelity); // inner loop length is the number of fidelity phases 0, 2, 3, 4 (LoFi, Hifi2, Hifi3, Hifi4)

    if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        ckernel_template tmp(1, inner_loop_len, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 4));
        tmp.set_last_inner_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.set_last_outer_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, inner_loop_len, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0));
        tmp.set_last_inner_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.set_last_outer_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.program();
    }
}

/**
 * @brief Configure the math (FPU) thread for a reduce operation: programs address mods and, for high fidelity, the pool MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam enforce_fp32_accumulation: Force FP32 accumulation (requires is_fp32_dest_acc_en).
 * @note @ref _llk_math_reduce_ runs the configured reduction with matching template args.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void _llk_math_reduce_init_()
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    reduce_configure_addrmod<type, math_fidelity>();
    if constexpr (high_fidelity)
    {
        reduce_configure_mop<dim, math_fidelity>();
    }

    if constexpr (enforce_fp32_accumulation)
    {
        static_assert(is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Uninitialize after a reduce operation, undoing any init/execute-time workarounds.
 *
 * @tparam enforce_fp32_accumulation: Must match the value used at init.
 * @note Reverses @ref _llk_math_reduce_init_; re-enables debug feature bit 11 (@ref _llk_math_dbg_feature_enable_) only when FP32 accumulation was enforced.
 */
template <bool enforce_fp32_accumulation = false>
inline void _llk_math_reduce_uninit_()
{
    if constexpr (enforce_fp32_accumulation)
    {
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
    }
}
