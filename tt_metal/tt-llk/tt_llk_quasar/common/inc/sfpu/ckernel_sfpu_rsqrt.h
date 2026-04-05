// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates RSQRT (reciprocal square root) for number of rows of output SFPU ops (Quasar = 2 rows)
// Always uses approximate mode (sqrt + recip) for optimal performance
inline void _calculate_rsqrt_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0]

    // First compute sqrt(x) into LREG1
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::SQRT_MODE); // Read value from lreg[0], sqrt, load back into lreg[1]

    // Then compute 1/sqrt(x) = recip(sqrt(x)) into LREG2
    TTI_SFPNONLINEAR(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpnonlinear::RECIP_MODE); // Read value from lreg[1], recip, load back into lreg[2]

    // Store from lreg[2] into dest register
    TTI_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

inline void _calculate_rsqrt_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_rsqrt_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
