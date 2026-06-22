// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"

namespace ckernel
{
namespace sfpu
{

// Int32 binary comparison for relational ops (signed), ported from BH.
// All ops reduce to computing LT(X, Y) with optional operand swap and result
// inversion:
//   lt(A,B) = LT(A,B)           gt(A,B) = LT(B,A)
//   ge(A,B) = NOT LT(A,B)       le(A,B) = NOT LT(B,A)
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_binary_comp_int32(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static_assert(
        RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le || RELATIONAL_OP == SfpuType::ge,
        "Supported operation types: lt, gt, le, ge");

    constexpr bool swap_operands = (RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le);
    constexpr bool invert_result = (RELATIONAL_OP == SfpuType::le || RELATIONAL_OP == SfpuType::ge);

    if constexpr (invert_result)
    {
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0x01);
    }

    const std::uint32_t idx_x = swap_operands ? dst_index_in1 : dst_index_in0;
    const std::uint32_t idx_y = swap_operands ? dst_index_in0 : dst_index_in1;

    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, idx_x + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, idx_y + (d << 1));

        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);

            // LT(X,Y): sign(X - Y); no overflow for Int8-promoted [-127,127] range.
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);

            if constexpr (invert_result)
            {
                TTI_SFPXOR(p_sfpu::LREG7, p_sfpu::LREG1);
            }

            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
        }
        else
        {
            TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(p_sfpu::LREG1, p_sfpu::LREG3, 0);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

            TTI_SFPXOR(p_sfpu::LREG2, p_sfpu::LREG3);

            // Same-sign path: subtract and extract sign bit.
            TTI_SFPSETCC(0, p_sfpu::LREG3, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);

            // Different-sign path: result = sign bit of X.
            TTI_SFPCOMPC;
            TTI_SFPMOV(p_sfpu::LREG2, p_sfpu::LREG1, 0);
            TTI_SFPENCC(0, 0);

            if constexpr (invert_result)
            {
                TTI_SFPXOR(p_sfpu::LREG7, p_sfpu::LREG1);
            }
        }

        TT_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_out + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
