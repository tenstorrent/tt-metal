// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-01_abs_quasar_779f878d

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates ABS for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_abs_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0]
    // Apply absolute value: clear sign bit for FP32 (instr_mod1=1)
    TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG0, 1);
    // Store result back to destination
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

inline void _calculate_abs_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_abs_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
