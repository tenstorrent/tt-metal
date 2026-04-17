// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
inline void _init_gelu_()
{
    // Need a fixed constant of 0.5 for the MAD part of the operation
    TTI_SFPLOADI(p_sfpu::LREG6, 0x8, 0x3F00);
    TTI_SFPLOADI(p_sfpu::LREG6, 0xA, 0x0000);

    // x >= 3.0
    // lreg11_hi =  0.50   (0x3800)
    // lreg14_hi =  0.0    (0x7c00)
    // 3.0 > x >= 2.0
    // lreg11_lo =  0.5402 (0x3852)
    // lreg14_lo = -0.1194 (0xAFA4)
    _sfpu_load_config32_(0xB, 0x3800, 0x3852);
    _sfpu_load_config32_(0xE, 0x7C00, 0xAFA4);

    // 2.0 > x >= 1.5
    // lreg10_hi =  0.6099 (0x38E1)
    // lreg13_hi = -0.2635 (0xB437)
    // 1.5 > x >= 1.0
    // lreg10_lo =  0.6189 (0x38F3)
    // lreg13_lo = -0.2797 (0xB479)
    _sfpu_load_config32_(0xA, 0x38E1, 0x38F3);
    _sfpu_load_config32_(0xD, 0xB437, 0xB479);

    // 1.0 > x >= 0.5
    // lreg9_hi  =  0.4939 (0x37E7)
    // lreg12_hi = -0.1605 (0xB122)
    // 0.5 > x >= 0.0
    // lreg9_lo  =  0.1928 (0x322B)
    // lreg12_lo = -0.0150 (0xA3AE)
    _sfpu_load_config32_(0x9, 0x37E7, 0x322B);
    _sfpu_load_config32_(0xC, 0xB122, 0xA3AE);
}

// Calculates GELU for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_gelu_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[3], uses ADDR_MOD_7 (set to all zeroes)

    TTI_SFPLUTFP32(p_sfpu::LREG4, 0x2); // Calculate piecewise part on lreg[3] and store in lreg[4], using FP16 6-entry format mode 1 LUT
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG5, 0); // 0.5 * x + piecewise result, store in lreg[5]
    TTI_SFPSTORE(p_sfpu::LREG5, 0, ADDR_MOD_7, 0, 0);                          // store from lreg[5] into dest register
}

inline void _calculate_gelu_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_gelu_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
