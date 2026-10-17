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
#include "lltt.h"
#include "sfpi.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

// local function declarations
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape);

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
    // The MOVD2B/ELWADD below read the Src zero substitution flag (FlushDenormals = !flag).
    // A datum whose low byte is zero (e.g. bf16 0x4400 = 768.0) would be flushed to 0 mid-reduction,
    // corrupting the sum. Disable the flag (via the math state tracker) around the transpose+add, then
    // return it to the operand driven baseline.
    math::_configure_preserve_zero_flag_state_();

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

    // Restore the operand-driven baseline for the currently configured formats.
    math::_configure_default_zero_flag_state_();
}

/**
 * @brief First half of the lossless fp32 REDUCE_SCALAR SUM/AVG tail: add the partial sums on the SFPU.
 *
 * The face pooling leaves the 16 per-column partial sums in DEST as fp32 datums (GAPOOL writes the aligned
 * 4-row block at rows 4-7, every row holding the same partial-sum row). The FPU tail then moves that row
 * through SrcB (MOVD2B, TRNSPSRCB) and SrcA (MOVB2A) for the final GAPOOL, and a Src datum holds 19 bits, so
 * each partial is truncated to TF32 mid-reduction and the total is low by the sum of those truncations even
 * though is_fp32_dest_acc_en asked for fp32.
 *
 * With fp32 DEST the partials never cross Src: this stage adds them on the SFPU in fp32, keeps the total in
 * lane 0 of LREG0, and rewrites the pooled 4-row block as row 4 = 0 and row 5 = [1.0, 0, ..., 0] (rows 6-7 = 0).
 * The FPU tail then moves all four rows to SrcB (MOVD2B MOV_4_ROWS, so SrcB row 16 + i is block row i) and the
 * transpose turns each of them into one result column: column 0 (row 4, all zero) stays the running accumulation
 * (+= 0, which also defines the row on the first tile of a reduce-to-one) and column 1 (row 5) becomes the
 * scaler operand (scaler * 1.0). @ref reduce_scalar_fp32_scale_total then computes total * scaler + accumulation
 * on the SFPU and writes it, so SUM/AVG are scaled and accumulated across tiles exactly as before.
 *
 * Must run after the last face pool and before the MOVD2B of the scratch row. Leaves the SFPU lane flags
 * enabled with all lanes on; LREG0 holds the total until @ref reduce_scalar_fp32_scale_total consumes it.
 */
inline void reduce_scalar_fp32_sum_partials()
{
    constexpr std::uint32_t SCRATCH_ROW = 4; // GAPOOL dst index used by the face pooling (rows 4-7)
    constexpr std::uint32_t ODD_COLS    = 2; // SFPLOAD/SFPSTORE address bit selecting the odd columns

    // Let the FPU pooling writes (including the HiFi MOP) land in DEST before the SFPU reads them.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    // Lane predication on with every lane enabled (the reduce kernel runs no SFPU init).
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);

    // Rows 4-7, even then odd columns: lane c of every 8-lane sub-vector gets partial[2c] and partial[2c + 1].
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_0, SCRATCH_ROW);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_0, SCRATCH_ROW + ODD_COLS);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    // Butterfly all-reduce over the 8 lanes of each sub-vector: rotate by 1, 2 and 4, adding each time.
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    // LREG0: every lane holds the total, summed in a lane-dependent order; only lane 0's value is used.

    // Pooled block := row 4 = 0, row 5 = [1.0, 0, ..., 0], rows 6-7 = 0: lane 8 of the even columns (row 5,
    // column 0) is 1.0 (LTILEID is 2 * lane, so LTILEID - 16 is zero only in lane 8), everything else is zero.
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPIADD(-16 & 0xFFF, p_sfpu::LTILEID, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_0, SCRATCH_ROW);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_0, SCRATCH_ROW + ODD_COLS);

    // Let the stores land before the FPU moves the row to SrcB.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

/**
 * @brief Second half of the lossless fp32 REDUCE_SCALAR SUM/AVG tail: scale the SFPU total, accumulate, write.
 *
 * Runs after the final GAPOOL, which left column 0 of the result row as the accumulation so far (zero on the
 * first tile) and column 1 as scaler * 1.0 (see @ref reduce_scalar_fp32_sum_partials). Computes
 * total * scaler + accumulation in fp32 from lane 0 of LREG0 and writes [result, 0, ..., 0] over the result
 * rows 0-3 (the packer reads element [0]; the zeroed columns keep column 1 clean for the next tile).
 */
inline void reduce_scalar_fp32_scale_total()
{
    constexpr std::uint32_t RESULT_ROW = 0;
    constexpr std::uint32_t ODD_COLS   = 2;

    // Let the final GAPOOL land in DEST before the SFPU reads the scaler from it.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    // Lane 0: even columns of row 0 give the accumulation (column 0), odd columns the scaler (column 1).
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_0, RESULT_ROW);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_0, RESULT_ROW + ODD_COLS);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG2, 0); // lane 0 = total * scaler + acc
    TTI_SFPNOP;

    // [result, 0, ..., 0] over rows 0-3: lane 0 of the even columns, zero elsewhere.
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPSETCC(0, p_sfpu::LTILEID, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_0, RESULT_ROW);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_0, RESULT_ROW + ODD_COLS);

    // Let the stores land before any later FPU write to this tile.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
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

/**
 * @brief Advance the dest counter past the current face row so the next row writes to the correct offset.
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
    TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_BD);
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

            lltt::record(replay_start, replay_buf_len);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(last_clr, 0, ADDR_MOD_3, 0);

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

            lltt::record<lltt::NoExec>(replay_start, replay_buf_len);
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
 * into a single column for a final pool (with fp32 DEST accumulation, SUM/AVG first add the partials on
 * the SFPU so the 19-bit Src transpose never carries them; see @ref reduce_scalar_fp32_sum_partials).
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
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool is_int_fpu_en = false>
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
            reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
            reduce_row_perform_transpose<is_int_fpu_en>();

            if (tensor_shape.num_faces_r_dim > 1)
            {
                reduce_row_advance_dest(is_narrow_tile);
                reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
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
        // SUM/AVG with fp32 DEST finish on the SFPU so the partial sums never cross the 19-bit Src registers.
        constexpr bool fp32_sfpu_tail = is_fp32_dest_acc_en && type != PoolType::MAX && !is_int_fpu_en;

        for (std::uint32_t face_num = 0; face_num < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face_num++)
        {
            // Wait and pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 4>();
        }
        // Wait and pool
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 4>();

        if constexpr (fp32_sfpu_tail)
        {
            reduce_scalar_fp32_sum_partials();
        }

        // Need row in dest as column in src A
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);

        // copy over from dest to B and do transpose
        // use rows 16 - 31 in src B as scratch (the fp32 tail moves its whole 4-row block, see
        // reduce_scalar_fp32_sum_partials)
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, fp32_sfpu_tail ? p_movd2b::MOV_4_ROWS : p_movd2b::MOV_1_ROW, 4);
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
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, 4);

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
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
                }
            }
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }

        if constexpr (fp32_sfpu_tail)
        {
            reduce_scalar_fp32_scale_total();
        }
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
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape)
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
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_reduce_init_(const ckernel::TensorShape& tensor_shape)
{
    // There is no min-pool instruction - the FPU has GMPOOL (max) and GAPOOL (average/sum).
    static_assert(
        type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    reduce_configure_addrmod<type, dim, math_fidelity>(tensor_shape);
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
inline void _llk_math_reduce_uninit_()
{
}
