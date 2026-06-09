// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

// Int32 multiply ported from the BH DISABLE_SFPLOADMACRO path.
// Uses SFPMUL24 (24-bit partial products) + shifts to produce a full 32-bit
// result.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _mul_int32_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG2, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));

        // Dest layout depends on how operands reached dest:
        //   UNP_DEST / Int32 L1 with 2's-comp tiles → 2's-comp Int32
        //   copy_tile Int8 + fp32_dest_acc FPU → sign-mag Int32 (SIGN_MAGNITUDE_FORMAT=true)
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
            TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
        }

        TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG0, p_sfpu::LREG1, 5); // lreg1 = lreg0 >> 23
        TTI_SFPSHFT((-23) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG3, 5); // lreg3 = lreg2 >> 23

        TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG4, 0); // low(a_lo * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG5, 1); // high(a_lo * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG6, 0); // low(a_hi * b_lo)
        TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG7, 0); // low(a_lo * b_hi)

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC); // lreg5 += lreg6
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC); // lreg5 += lreg7
        TTI_SFPSHFT(23, p_sfpu::LREG5, p_sfpu::LREG5, 5);                                         // lreg5 <<= 23
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC); // lreg4 += lreg5

        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
        }

        TT_SFPSTORE(p_sfpu::LREG4, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_out + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
