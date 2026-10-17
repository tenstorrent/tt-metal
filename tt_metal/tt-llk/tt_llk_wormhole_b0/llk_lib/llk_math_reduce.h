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

// REDUCE_SCALAR pools every face into this DEST row (GAPOOL writes the aligned 4-row block starting here) before
// the final transpose + pool; the fp32 SUM/AVG tail addresses the same block.
constexpr std::uint32_t REDUCE_SCALAR_SCRATCH_ROW = 4;
// SFPLOAD/SFPSTORE address bit that selects the odd columns of a 4-row block.
constexpr std::uint32_t SFPU_ODD_COLS = 2;
// The fp32 SUM/AVG tail pools a second copy of its pieces into this 4-row block (cleared right before) so the
// mid/lo sums never mix with whatever the result row already holds.
constexpr std::uint32_t REDUCE_SCALAR_PIECE_SUM_ROW = 8;
// SFPSHFT Mod1: the shift amount is the immediate (ISA SFPSHFT_MOD1_ARG_IMM).
constexpr std::uint32_t SFPSHFT_MOD1_ARG_IMM = 1;
// Position of the fp32 exponent field.
constexpr std::uint32_t FP32_EXPONENT_SHIFT = 23;
// The fp32 SUM/AVG tail splits only partials with at least this biased exponent (2^-63 and up).
constexpr std::uint32_t REDUCE_SCALAR_SPLIT_MIN_BIASED_EXPONENT = 64;

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
 * @brief First half of the fp32-DEST REDUCE_SCALAR SUM/AVG tail: split each fp32 partial sum into Src-exact pieces.
 *
 * The face pooling leaves the 16 per-column partial sums in DEST as fp32 datums (GAPOOL writes the aligned 4-row
 * block at REDUCE_SCALAR_SCRATCH_ROW). The FPU tail then moves that row through SrcB (MOVD2B, TRNSPSRCB) and
 * SrcA (MOVB2A) for the final GAPOOL, and a Src datum holds 19 bits, so each partial is truncated to TF32
 * mid-reduction and the total is low by the sum of those truncations even though is_fp32_dest_acc_en asked
 * for fp32 (the Float32 input is still TF32 in SrcA for the face pooling; what this tail keeps at fp32 is the
 * partial sums).
 *
 * With fp32 DEST and HiFi2 or higher each partial P is split on the SFPU into P = hi + mid + lo, every piece
 * exact in 8 significant bits (hi = P & 0xFFFF0000, the bf16 truncation; mid likewise of P - hi; lo the
 * remainder), which a BF16 or TF32 Src datum carries without loss and the multiplier consumes in its first two
 * fidelity phases (under LoFi it would consume 4 mantissa bits of hi and nothing of mid or lo). The block
 * is rewritten as row 0 = hi, row 1 = mid, row 2 = lo (row 3 unused), and the FPU tail moves those three rows
 * to SrcB rows 16, 30 and 31, so after the transpose the final GAPOOL weights every piece by its own column of
 * the scaler row, exactly as it weighted the whole partials before (a non-uniform scaler row keeps its meaning),
 * and lands the sums in columns 0 (hi, on top of the running accumulation of the result row), 14 (mid) and
 * 15 (lo); a second copy of that pool into the piece-sum rows, cleared right before it, provides the mid and lo
 * sums free of whatever the result row held, and @ref reduce_scalar_fp32_sum_pieces adds the three in fp32.
 * A non-finite partial stays whole in hi with mid = lo = 0, and so does a partial below 2^-63, whose pieces could
 * fall below 2^-126 (it keeps the pre-fix precision: TF32 or BF16 in Src).
 *
 * Runs after the last face pool and before the MOVD2B of the scratch block. Overwrites SFPU LREG0-LREG7 and leaves
 * the lane flags enabled with all lanes on (callers
 * re-run any SFPU init that preloads an LREG). Kept out of line like _apply_src_zero_flag_: a kernel with several
 * REDUCE_SCALAR SUM sites would otherwise carry one copy of this sequence per site.
 */
inline __attribute__((noinline)) void reduce_scalar_fp32_split_partials()
{
    // Upper halves of the piece mask 0xFFFF0000 (sign, exponent and the 7 mantissa bits a bf16 keeps) and of the
    // exponent mask 0x7F800000; SFPLOADI FLOATB writes Imm16 << 16.
    constexpr std::uint32_t PIECE_MASK    = 0xFFFF;
    constexpr std::uint32_t EXPONENT_MASK = 0x7F80;

    // Let the FPU pooling writes (including the HiFi MOP) land in DEST before the SFPU reads them.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    // Even columns of the block into LREG0, odd columns into LREG4. Only lane group 0 (block row 0, the
    // partial-sum row) matters: the SFPTRANSP below gathers lane group 0 of LREG0-2 / LREG4-6, so the pieces
    // computed on lane groups 1-3 (the other block rows) are never stored.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_ROW);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_ROW + SFPU_ODD_COLS);

    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, PIECE_MASK); // Imm16 << 16, no conversion

    // hi = P & mask (LREG0 / LREG4), keeping P in LREG1 / LREG5.
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG1, 0 /*instr_mod1*/);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG4, p_sfpu::LREG5, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG0, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG4, 0 /*instr_mod1*/);
    // lo = P - hi (LREG2 / LREG6), exact.
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPMOV_MOD1_COMPSIGN);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG4, p_sfpu::LREG6, sfpi::SFPMOV_MOD1_COMPSIGN);
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 0 /*instr_mod1*/);
    TTI_SFPADD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG6, 0 /*instr_mod1*/);
    // mid = lo & mask (LREG1 / LREG5); each SFPADD result is read two instructions later, so no NOP is needed.
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG2, p_sfpu::LREG1, 0 /*instr_mod1*/);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG6, p_sfpu::LREG5, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG1, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG5, 0 /*instr_mod1*/);
    // lo = lo - mid (LREG2 / LREG6), exact; the mask in LREG3 is dead from here on.
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG7, sfpi::SFPMOV_MOD1_COMPSIGN);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG5, p_sfpu::LREG3, sfpi::SFPMOV_MOD1_COMPSIGN);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG2, 0 /*instr_mod1*/);
    TTI_SFPADD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG6, 0 /*instr_mod1*/);

    // Non-finite partials (an fp32 overflow during the face pooling leaves an exponent-255 pattern) must not go
    // through the residual subtractions: inf - inf is NaN on the SFPU, and the FPU reads exponent-255 Src patterns
    // numerically. Keep such a partial in hi (the mask preserves its exponent and sign) and zero its mid and lo.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, EXPONENT_MASK);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);
    // even columns: LREG7 = mask - (hi & mask), zero exactly when the exponent is all ones
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPIADD(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG7, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0 /*imm12_math*/, p_sfpu::LREG7, 0 /*lreg_dest*/, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG1, 0 /*instr_mod1*/);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG2, 0 /*instr_mod1*/);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);
    // odd columns
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG4, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPIADD(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG7, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0 /*imm12_math*/, p_sfpu::LREG7, 0 /*lreg_dest*/, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG5, 0 /*instr_mod1*/);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG6, 0 /*instr_mod1*/);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);

    // Partials below 2^-63 (biased exponent < REDUCE_SCALAR_SPLIT_MIN_BIASED_EXPONENT) are not split: their pieces
    // can reach below 2^-126, where the SFPU flushes the residual and the FPU the Src datum to zero, so a split
    // would lose bits the whole partial keeps. They go through whole (reloaded from DEST, mid = lo = 0), which is
    // the pre-fix precision; above the threshold every piece times any scaler down to 2^-40 stays normal.
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPSHFT((-FP32_EXPONENT_SHIFT) & 0xFFF, 0 /*lreg_c*/, p_sfpu::LREG7, SFPSHFT_MOD1_ARG_IMM);
    TTI_SFPIADD((-REDUCE_SCALAR_SPLIT_MIN_BIASED_EXPONENT) & 0xFFF, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPIADD_MOD1_ARG_IMM);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_ROW);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG1, 0 /*instr_mod1*/);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG2, 0 /*instr_mod1*/);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);
    // odd columns
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LREG4, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPAND(0 /*imm12_math*/, p_sfpu::LREG3, p_sfpu::LREG7, 0 /*instr_mod1*/);
    TTI_SFPSHFT((-FP32_EXPONENT_SHIFT) & 0xFFF, 0 /*lreg_c*/, p_sfpu::LREG7, SFPSHFT_MOD1_ARG_IMM);
    TTI_SFPIADD((-REDUCE_SCALAR_SPLIT_MIN_BIASED_EXPONENT) & 0xFFF, p_sfpu::LREG7, p_sfpu::LREG7, sfpi::SFPIADD_MOD1_ARG_IMM);
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_ROW + SFPU_ODD_COLS);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG5, 0 /*instr_mod1*/);
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG6, 0 /*instr_mod1*/);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);

    // Transpose: LREG0 lane groups 0-2 = lane group 0 of hi, mid, lo of the even columns (LREG0-2), LREG4 the same
    // for the odd columns (LREG4-6); lane group 3 (block row 3) carries LREG3 / LREG7 and is never moved.
    TTI_SFPTRANSP(0 /*imm12_math*/, 0 /*lreg_c*/, 0 /*lreg_dest*/, 0 /*instr_mod1*/);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_ROW);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_ROW + SFPU_ODD_COLS);

    // Let the stores land before the FPU moves the block to SrcB.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

/**
 * @brief Second half of the fp32-DEST REDUCE_SCALAR SUM/AVG tail: add the three piece sums and write the result.
 *
 * Runs after the two final GAPOOLs (see @ref reduce_scalar_fp32_split_partials): the result rows hold the
 * accumulation so far plus the scaler-weighted hi pieces in column 0 (the FPU reads an undefined result row as
 * zero, so a fresh DEST section and a caller-preloaded accumulation both work as before), and the piece-sum rows,
 * cleared right before that pool, hold the weighted mid pieces in column 14 and the weighted lo pieces in column
 * 15. Adds the three in fp32 and writes column 0 of the result rows (the packer reads element [0]; the other
 * columns of the result rows are left exactly as the FPU tail leaves them). Overwrites SFPU LREG0-LREG3 and
 * leaves the lane flags enabled with all lanes on.
 */
inline __attribute__((noinline)) void reduce_scalar_fp32_sum_pieces()
{
    constexpr std::uint32_t RESULT_ROW         = 0;
    constexpr std::uint32_t LTILEID_COL0_SHIFT = 28; // LTILEID << 28 is zero only in lanes 0, 8, 16, 24 (column 0 of each row)

    // Let the final GAPOOLs land in DEST before the SFPU reads them.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    // Even columns of the result rows: lane 0 (and 8, 16, 24) = column 0. Piece-sum rows: lane 7 of the even
    // columns = column 14, lane 7 of the odd columns = column 15; one rotation brings both to lane 0 (8, 16, 24).
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_0, RESULT_ROW);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_PIECE_SUM_ROW);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_0, REDUCE_SCALAR_PIECE_SUM_ROW + SFPU_ODD_COLS);
    // Wormhole only forbids the very next instruction from reading a SUBVEC_SHFLROR1 result, so the two rotations
    // issue back to back and the first SFPADD follows them directly (same spacing as horizontal_reduce).
    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
    // Store column 0 of the result rows only (the SFPENCC fills the SFPADD-to-SFPADD slot; LREG0 is next read by
    // the SFPSTORE four instructions after the last SFPADD).
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*instr_mod1*/); // column 0 lanes = result
    TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LTILEID, p_sfpu::LREG3, 0 /*instr_mod1*/);
    TTI_SFPSHFT(LTILEID_COL0_SHIFT, 0 /*lreg_c*/, p_sfpu::LREG3, SFPSHFT_MOD1_ARG_IMM);
    TTI_SFPSETCC(0 /*imm12_math*/, p_sfpu::LREG3, 0 /*lreg_dest*/, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_0, RESULT_ROW);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0 /*lreg_c*/, 0 /*lreg_dest*/, sfpi::SFPENCC_MOD1_EI_RI);

    // Let the store land before any later FPU write to this tile.
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
        constexpr std::uint32_t dst  = (dim == ReduceDim::REDUCE_SCALAR) ? REDUCE_SCALAR_SCRATCH_ROW : 0;

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
 * into a single column for a final pool (with fp32 DEST accumulation and HiFi2 or higher, SUM/AVG first split
 * the partials into Src-exact pieces on the SFPU so the 19-bit Src transpose keeps them at fp32 precision, and
 * overwrite SFPU LREG0-LREG7 doing so; see @ref reduce_scalar_fp32_split_partials).
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
        // SUM/AVG with fp32 DEST at HiFi2 and up split the partial sums into Src-exact pieces on the SFPU (see
        // reduce_scalar_fp32_split_partials); MAX, LoFi (SrcA contributes 4 mantissa bits, the pieces would gain
        // nothing), 16-bit DEST and int keep the plain transpose.
        constexpr bool fp32_sfpu_tail = is_fp32_dest_acc_en && high_fidelity && type != PoolType::MAX && !is_int_fpu_en;

        for (std::uint32_t face_num = 0; face_num < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face_num++)
        {
            // Wait and pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, REDUCE_SCALAR_SCRATCH_ROW>();
        }
        // Wait and pool
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, REDUCE_SCALAR_SCRATCH_ROW>();

        if constexpr (fp32_sfpu_tail)
        {
            reduce_scalar_fp32_split_partials();
        }

        // Need row in dest as column in src A
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);

        // copy over from dest to B and do transpose
        // use rows 16 - 31 in src B as scratch
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, REDUCE_SCALAR_SCRATCH_ROW);
        if constexpr (fp32_sfpu_tail)
        {
            // The mid and lo pieces go to SrcB rows 30 and 31, i.e. result columns 14 and 15 after the transpose.
            TTI_MOVD2B(0 /*dest_32b_lo*/, p_movd2b::SRC_ROW16_OFFSET + 14, ADDR_MOD_0, p_movd2b::MOV_1_ROW, REDUCE_SCALAR_SCRATCH_ROW + 1);
            TTI_MOVD2B(0 /*dest_32b_lo*/, p_movd2b::SRC_ROW16_OFFSET + 15, ADDR_MOD_0, p_movd2b::MOV_1_ROW, REDUCE_SCALAR_SCRATCH_ROW + 2);
        }
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
        // zero out scratch in dest: only the pooled row (block row 0) is cleared, so the next tile's pooling starts
        // from zero there; with the fp32 tail block rows 1-2 keep the mid/lo pieces, which is fine because that
        // tail re-derives every row from block row 0 and the packer keeps element [0] only
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_ROW);
        if constexpr (fp32_sfpu_tail)
        {
            // Clear the piece-sum rows so the second pool below starts from zero whatever they (or a preloaded result
            // row) hold.
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, REDUCE_SCALAR_PIECE_SUM_ROW + 0);
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, REDUCE_SCALAR_PIECE_SUM_ROW + 1);
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, REDUCE_SCALAR_PIECE_SUM_ROW + 2);
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, ADDR_MOD_0, REDUCE_SCALAR_PIECE_SUM_ROW + 3);
        }

        if constexpr (type == PoolType::MAX)
        {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
        else
        {
            if constexpr (fp32_sfpu_tail)
            {
                // Same pool into the cleared piece-sum rows: their columns 14 and 15 are the mid and lo sums alone.
                if constexpr (high_fidelity)
                {
                    for (std::uint32_t i = 0; i < to_underlying(math_fidelity) - 1; i++)
                    {
                        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, REDUCE_SCALAR_PIECE_SUM_ROW);
                    }
                }
                TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, REDUCE_SCALAR_PIECE_SUM_ROW);
            }
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
            reduce_scalar_fp32_sum_pieces();
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
