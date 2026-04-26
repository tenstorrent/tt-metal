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
inline void _calculate_add_(const DataFormat fmt, const int iterations, const int in0_offset_idx, const int in1_offset_idx, const int out_offset_idx)
{
    const bool is_int    = (fmt == DataFormat::Int32 || fmt == DataFormat::Int16);
    const auto instr_mod = is_int ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT;

    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod, ADDR_MOD_7, 0, in0_offset_idx + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, in1_offset_idx + (d << 1));

        if (is_int)
        {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0x3); // S+M -> 2SC
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0x3); // S+M -> 2SC

            TTI_SFPIADD(0x0, p_sfpu::LREG0, p_sfpu::LREG1, 0b0100);

            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0x2); // 2SC -> S+M

            TT_SFPSTORE(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, out_offset_idx + (d << 1));
        }
        else
        {
            TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);

            TT_SFPSTORE(p_sfpu::LREG2, instr_mod, ADDR_MOD_7, 0, out_offset_idx + (d << 1));
        }
    }
}

} // namespace sfpu
} // namespace ckernel
