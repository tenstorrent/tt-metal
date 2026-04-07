// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
inline void _calculate_add_(const int iterations, const int in0_offset_idx, const int in1_offset_idx, const int out_offset_idx)
{
    for (int d = 0; d < iterations; d++)
    {
        // Load inputs into LREG0 and LREG1, if want to load from dest, offset should start from 0x0
        // if want to load from SrcS, inputs should start from 0x400
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in0_offset_idx + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in1_offset_idx + (d << 1));

        // Do LREG0 + LREG1, store result in LREG2, takes 2 cycles
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);
        TTI_NOP;
        // Store result from LREG2 into dest, takes 2 cycles
        TT_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_offset_idx + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
