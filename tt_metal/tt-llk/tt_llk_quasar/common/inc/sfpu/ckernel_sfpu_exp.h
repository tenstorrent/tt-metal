// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates EXP for one pair of SrcS rows (Quasar SFPU ops cover 2 rows)
inline void _calculate_exp_srcs_rows_(const int load_addr, const int store_addr)
{
    TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, load_addr);   // load from SrcS into lreg[0]
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::EXP_MODE);       // lreg[1] = exp(lreg[0])
    TT_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, store_addr); // store from lreg[1] into SrcS
}

// Implements element-wise exponential exp(x) on SrcS: reads num_sfpu_iterations row pairs
// starting at load_base_addr and writes the results starting at store_base_addr.
inline void _calculate_exp_srcs_(const int load_base_addr, const int store_base_addr, const int num_sfpu_iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < num_sfpu_iterations; d++)
    {
        _calculate_exp_srcs_rows_(load_base_addr + (d << 1), store_base_addr + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
