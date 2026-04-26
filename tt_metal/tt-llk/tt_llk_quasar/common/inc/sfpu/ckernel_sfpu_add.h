// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
template <bool INT_EN = true>
inline void _calculate_add_(const int iterations, const int in0_offset_idx, const int in1_offset_idx, const int out_offset_idx)
{
    for (int d = 0; d < iterations; d++)
    {
        // Load inputs into LREG0 and LREG1, if want to load from dest, offset should start from 0x0
        // if want to load from SrcS, inputs should start from 0x400
        constexpr auto INSTR_MOD = INT_EN ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT;
        TT_SFPLOAD(p_sfpu::LREG0, INSTR_MOD, ADDR_MOD_7, 0, in0_offset_idx + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, INSTR_MOD, ADDR_MOD_7, 0, in1_offset_idx + (d << 1));

        if constexpr (INT_EN)
        {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0x3); // convert sign magnitude to 2s complement
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0x3); // convert sign magnitude to 2s complement

            // Do LREG1 = LREG0 + LREG1, takes 1 cycle
            TTI_SFPIADD(0x0, p_sfpu::LREG0, p_sfpu::LREG1, 0b0100);

            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0x2); // convert back to sign magnitude
        }
        else
        {
            TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);
        }

        constexpr auto LREG_DEST = INT_EN ? p_sfpu::LREG1 : p_sfpu::LREG2;
        TT_SFPSTORE(LREG_DEST, INSTR_MOD, ADDR_MOD_7, 0, out_offset_idx + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
