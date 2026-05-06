// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"

namespace ckernel
{
namespace sfpu
{
inline void _calculate_add_(const DataFormat fmt, const int iterations, const int in0_offset_idx, const int in1_offset_idx, const int out_offset_idx)
{
    LLK_ASSERT(fmt == DataFormat::Int32 || fmt == DataFormat::Float16_b, "Only Int32 and Float16_b are currently supported for SFPU add on Quasar");

    const bool is_int = (fmt == DataFormat::Int32);
    const auto instr_mod = is_int ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT; // There is a quasar bug with implied fmts + upk to dest, so we need use
                                                                                     // use explicit types for int SFPULOAD/STORE TEN-4674

    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod, ADDR_MOD_7, 0, in0_offset_idx + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, in1_offset_idx + (d << 1));

        if (is_int)
        {
            // On Quasar, SFPU kernels should assume that integer inputs are in 2's complement format
            // TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC); // Sign+Mag -> 2SC
            // TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC); // Sign+Mag-> 2SC

            TTI_SFPIADD(
                0x0,
                p_sfpu::LREG0,
                p_sfpu::LREG1,
                p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC); // SFPIADD needs to explicitly disable CC output since CC exu is enabled by default

            // TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM); // 2SC -> Sing+Mag

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
