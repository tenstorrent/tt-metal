// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates SILU for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_silu_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    // Calculate sigmoid using lreg[0] as src, lreg[1] as work register, and lreg[2] as dest (since we need the original value for the final multiply)
    _calculate_sigmoid_regs_(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG1, 0); // Multiply lreg[0] * lreg[2], store result in lreg[1]

    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 0); // store from lreg[1] into dest register
}

inline void _calculate_silu_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_silu_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
