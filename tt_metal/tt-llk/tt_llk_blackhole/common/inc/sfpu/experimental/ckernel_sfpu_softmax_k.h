// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool is_fp32_dest_acc_en>
inline void _init_softmax_k_()
{
    sfpu::exp_init<false, 0x3F800000, true, is_fp32_dest_acc_en>();
}

// lreg = 2 * (L % 8) in lane L: the DEST column an address-0 SFPLOAD puts in that lane.
//
// Lane L of an SFPLOAD at address 0 is DEST column 2*(L%8) and at address 2 column 2*(L%8)+1 (SFPLOAD.md), for a
// 16-bit and a 32-bit DEST alike, and LTILEID (LReg[15]) holds 2*L in lane L, so its low four bits are 2*(L%8).
template <std::uint32_t lreg>
inline void _even_column_index_()
{
    // SFPSHFT Mod1 bits (SFPSHFT.md): shift by Imm12 instead of by LReg[VC]; with it, shift LReg[VC] into VD.
    constexpr std::uint32_t SFPSHFT_MOD1_ARG_IMM        = 1;
    constexpr std::uint32_t SFPSHFT_MOD1_ARG_IMM_USE_VC = 4;
    // Imm12 is signed (negative shifts right) and is masked to its 12 bits: -28 << 12 would carry into the opcode.
    constexpr std::uint32_t shift_left_28  = 28;
    constexpr std::uint32_t shift_right_28 = (-28) & 0xFFF;
    // Shift LTILEID's low four bits to the top, then logically back down.
    TTI_SFPSHFT(shift_left_28, p_sfpu::LTILEID, lreg, SFPSHFT_MOD1_ARG_IMM | SFPSHFT_MOD1_ARG_IMM_USE_VC);
    TTI_SFPSHFT(shift_right_28, 0 /*lreg_c*/, lreg, SFPSHFT_MOD1_ARG_IMM);
}

// LREG0 = the sum of the k valid columns of each row, in every lane of that row. On entry LREG0 holds the even
// columns and LREG1 the odd columns (SFPLOAD at addresses 0 and 2). Clobbers LREG1 to LREG3; leaves every lane
// enabled.
template <int k>
inline void _row_sum_across_lanes_()
{
    constexpr int face_c_dim = static_cast<int>(FACE_C_DIM);

    // 1. Pair add.
    if constexpr (k == face_c_dim)
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
        TTI_SFPNOP; // the next SFPSHFT2 reads LREG0
    }
    else if constexpr (k % 2 == 0)
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
        _even_column_index_<p_sfpu::LREG2>();
        // Both columns of instance L%8 are padding iff 2*(L%8) - k >= 0.
        TTI_SFPIADD((-k) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
        TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*instr_mod1*/);
        TTI_SFPENCC(0 /*imm12_math*/, 0 /*lreg_c*/, 0 /*lreg_dest*/, 0 /*instr_mod1*/);
    }
    else
    {
        _even_column_index_<p_sfpu::LREG2>();
        // Odd column 2*(L%8)+1 is padding iff 2*(L%8) - (k - 1) >= 0.
        TTI_SFPIADD((-(k - 1)) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
        TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG1, 0 /*instr_mod1*/);
        if constexpr (k < face_c_dim - 1)
        {
            // Even column 2*(L%8) is padding iff 2*(L%8) - k >= 0 (k = 15 leaves no even padding).
            TTI_SFPIADD((-k) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*instr_mod1*/);
        }
        TTI_SFPENCC(0 /*imm12_math*/, 0 /*lreg_c*/, 0 /*lreg_dest*/, 0 /*instr_mod1*/);
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
        TTI_SFPNOP; // the next SFPSHFT2 reads LREG0
    }

    // 2. Butterfly sum. Each lane result is bit identical.
    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
    TTI_SFPNOP; // the next SFPSHFT2 reads LREG0

    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
    TTI_SFPNOP; // the next SFPSHFT2 reads LREG0

    TTI_SFPSHFT2(0 /*imm12_math*/, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /*instr_mod1*/);
}

// Zero the padding columns (>= k) of the normalized result before it is stored: LREG1 holds the even columns and
// LREG2 the odd columns, LREG4 the even column index from _even_column_index_. The odd lane is padding when
// 2*(L%8) >= k - 1 and the even lane when 2*(L%8) >= k; the second set is inside the first, so the even test is
// nested. Clobbers LREG3; leaves every lane enabled.
template <int k>
inline void _zero_padding_columns_of_output_()
{
    constexpr int face_c_dim = static_cast<int>(FACE_C_DIM);
    if constexpr (k < face_c_dim)
    {
        TTI_SFPIADD((-(k - 1)) & 0xFFF, p_sfpu::LREG4, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
        TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG2, 0 /*instr_mod1*/);
        if constexpr (k < face_c_dim - 1)
        {
            TTI_SFPIADD((-k) & 0xFFF, p_sfpu::LREG4, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_GTE0);
            TTI_SFPMOV(0 /*imm12_math*/, p_sfpu::LCONST_0, p_sfpu::LREG1, 0 /*instr_mod1*/);
        }
        TTI_SFPENCC(0 /*imm12_math*/, 0 /*lreg_c*/, 0 /*lreg_dest*/, 0 /*instr_mod1*/);
    }
}

/**
 * @brief Per-row softmax over the first k columns of face 0's first row band (DEST rows 0-3).
 *
 * Reads x from DEST rows 0-3 (columns 0-15) and max(x) per row, pre-broadcast across the row, from DEST rows 8-11.
 * Writes out[r][c] = exp(x[r][c] - m[r]) / sum_{c' < k} exp(x[r][c'] - m[r]) back over rows 0-3 for c < k.
 *
 * Padding (columns k..15 of rows 0-3):
 * - Input: Ignored.
 * - Output: 0.0.
 *
 * DEST rows 8-11 (the maxima) and every row outside 0-3 / 8-11 are left unchanged.
 *
 * @tparam k: number of valid columns per row, 1..16 (enforced by static_assert).
 * @tparam is_fp32_dest_acc_en: DEST width; selects the bf16 or the fp32 exponential.
 * @note Run @ref _init_softmax_k_ first. Call with VectorMode::RC_custom: the kernel addresses DEST itself.
 * @note Every SFPU lane must be enabled on entry (lane flags all set, as SFPU kernels leave them). The kernel leaves
 *       every lane enabled. No general-purpose LREG survives the call.
 */
template <int k, bool is_fp32_dest_acc_en>
inline void _softmax_k_()
{
    static_assert(k >= 1 && k <= static_cast<int>(FACE_C_DIM), "softmax_k: k must be in 1..16");

    // LREG0 = x - max(x)
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 8);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0);

    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);

    // exp(x - max(x)) over every lane, padding included (its value is discarded below)
    sfpu::calculate_exponential<
        false,
        is_fp32_dest_acc_en,
        false, // scaling
        2,     // iterations
        true   // clamp negatives
        >();

    math::clear_dst_reg_addr();

    // LREG0 = sum over the k valid columns of exp(x - max(x)), in every column of its row.
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 2);
    _row_sum_across_lanes_<k>();
    if constexpr (k < static_cast<int>(FACE_C_DIM))
    {
        // For the output mask below; also fills the SFPADD latency before SFPARECIP reads LREG0.
        _even_column_index_<p_sfpu::LREG4>();
    }

    // LREG0 = 1 / sum(exp(x - max(x))).
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    // Normalize both column groups, zero the padding columns and write the completed softmax to rows 0-3.
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    _zero_padding_columns_of_output_<k>();
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 2);
}

} // namespace sfpu
} // namespace ckernel
