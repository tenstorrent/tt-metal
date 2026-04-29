// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// imm12 bit 11 = 1: SFPSETCC interprets src_c as two's-complement INT32, not FP32/SMAG32
constexpr std::uint32_t SFPSETCC_INT32_SIGNBIT = 0x800;

// Float variant — inner row body.
// Loads two FP rows from separate Dest tile regions, computes element-wise
// min+max via SFPSWAP(mod1=1), then stores the result for IS_MAX_OP.
//
// SFPLOAD/SFPSTORE use sfpmem::DEFAULT (MOD0_FMT_SRCB) — SFPLOAD resolves it
// at runtime via ALU_ACC_CTRL_SFPU_Fp32_enabled and the SrcB format register,
// both of which the unpacker / _llk_math_srcAB_hw_configure_ already program
// from formats.math. So DEFAULT lands on FP16A / FP16B / FP32 to match Dst.
//
// CC-guarded correction swap: SFPSWAP_VEC_MIN_MAX on Quasar uses an unsigned
// 32-bit compare on the LReg bits instead of the documented sign-magnitude
// (SignMagIsSmaller) compare. For (neg, neg) pairs that inverts the ordering —
// the same bug the int32 variant works around below. Sign-magnitude FP32 in
// LReg has bit 31 = sign just like INT32, so the same SFPSETCC-on-LT0 trick
// applies: detect rows where both LRegs are negative and re-swap them.
//
// @param offset0    Base Dest address (in 16b units) of the in0 tile region.
// @param offset1    Base Dest address (in 16b units) of the in1 tile region.
// @param offset2    Base Dest address (in 16b units) of the output tile region.
// @param row_index  Row index within the tile [0, ITERATIONS); selects which of the 32-lane SFPU rows to process.
template <bool IS_MAX_OP = true>
inline void _calculate_binary_max_min_sfp_rows_(const std::uint32_t offset0, const std::uint32_t offset1, const std::uint32_t offset2, const int row_index)
{
    TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, offset0 + (row_index << 1)); // load FP row from in0
    TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /* done */, offset1 + (row_index << 1)); // load FP row from in1

    // Step 1: SM-min/SM-max via SFPSWAP. Correct for any pair where at least one
    // operand is non-negative; inverts ordering for (neg, neg) pairs.
    TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // 2-cycle
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);                // post-SFPSWAP stall avoidance

    // Step 2: CC-guarded correction swap for (neg, neg) pairs. SFPSETCC with
    // imm12=SFPSETCC_INT32_SIGNBIT interprets the LReg as INT32 and tests bit 31,
    // which is the sign bit for both INT32 SM and FP32 SM. Successive SFPSETCC
    // calls AND into CC, so after both we have CC = (LREG0<0 AND LREG1<0).
    TTI_SFPSETCC(SFPSETCC_INT32_SIGNBIT, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(SFPSETCC_INT32_SIGNBIT, p_sfpu::LREG1, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP); // re-swap rows where both operands are negative
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);

    // After step 2: LREG0 = min, LREG1 = max for all sign combinations.
    TT_SFPSTORE(
        IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0 /* done */,
        offset2 + (row_index << 1)); // store max (LREG1) or min (LREG0)
}

// Float variant — outer loop.
// Explicit offset arithmetic for all three tile regions; no _incr_counters_
// needed because loads/stores use absolute addresses, not the auto-increment pointer.
//
// @param dst_index_in0  Dest tile index of input 0 (in tile units).
// @param dst_index_in1  Dest tile index of input 1 (in tile units).
// @param dst_index_out  Dest tile index where the result tile is written (in tile units).
template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    const std::uint32_t offset0 = (dst_index_in0 * 32) << 1;
    const std::uint32_t offset1 = (dst_index_in1 * 32) << 1;
    const std::uint32_t offset2 = (dst_index_out * 32) << 1;
#pragma GCC unroll 8
    for (int row_index = 0; row_index < ITERATIONS; row_index++)
    {
        _calculate_binary_max_min_sfp_rows_<IS_MAX_OP>(offset0, offset1, offset2, row_index);
    }
}

// Int32 variant — inner row body.
// SFPSWAP mod1=1 (VEC_MIN_MAX) does sign-magnitude compare, which inverts
// ordering for two's-complement (neg, neg) pairs. The CC-guarded correction
// swap re-fixes those rows. The correction is a no-op for unsigned-origin
// LRegs (bit 31 always 0), so this body also works for any integer source
// format that the upstream math path has rebased into INT32 sign-mag (e.g.
// UInt8 via EN_INT32_MATH_FORMAT).
//
// @param offset0    Base Dest address (in 16b units) of the in0 tile region.
// @param offset1    Base Dest address (in 16b units) of the in1 tile region.
// @param offset2    Base Dest address (in 16b units) of the output tile region.
// @param row_index  Row index within the tile; selects which of the 32-lane SFPU rows to process.
template <bool IS_MAX_OP = true>
inline void _calculate_binary_max_min_int32_sfp_rows_(
    const std::uint32_t offset0, const std::uint32_t offset1, const std::uint32_t offset2, const int row_index)
{
    TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, offset0 + (row_index << 1)); // load INT32 row from in0
    TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, offset1 + (row_index << 1)); // load INT32 row from in1

    // Step 1: sign-magnitude min/max — VD=LREG0 gets SM-min, VC=LREG1 gets SM-max.
    TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // 2-cycle
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);                // post-SFPSWAP stall avoidance

    // Step 2: CC-guarded correction swap for two's-complement (neg, neg) pairs.
    // imm12=SFPSETCC_INT32_SIGNBIT tells SFPSETCC to interpret src_c as INT32.
    TTI_SFPSETCC(SFPSETCC_INT32_SIGNBIT, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0); // CC = LREG0<0
    TTI_SFPSETCC(SFPSETCC_INT32_SIGNBIT, p_sfpu::LREG1, sfpi::SFPSETCC_MOD1_LREG_LT0); // CC &= LREG1<0
    TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP); // re-swap rows where both negative
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);

    TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, offset2 + (row_index << 1)); // store INT32 result
}

// Int32 variant — outer loop.
//
// @param dst_index_in0  Dest tile index of input 0 (in tile units).
// @param dst_index_in1  Dest tile index of input 1 (in tile units).
// @param dst_index_out  Dest tile index where the result tile is written (in tile units).
template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    const std::uint32_t offset0 = (dst_index_in0 * 32) << 1;
    const std::uint32_t offset1 = (dst_index_in1 * 32) << 1;
    const std::uint32_t offset2 = (dst_index_out * 32) << 1;
#pragma GCC unroll 8
    for (int row_index = 0; row_index < ITERATIONS; row_index++)
    {
        _calculate_binary_max_min_int32_sfp_rows_<IS_MAX_OP>(offset0, offset1, offset2, row_index);
    }
}

} // namespace sfpu
} // namespace ckernel
