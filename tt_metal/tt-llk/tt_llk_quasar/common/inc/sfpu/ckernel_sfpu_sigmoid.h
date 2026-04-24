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
// Calculates SIGMOID for results already in registers, used as a helper function.
// Input of sigmoid is expected to already be in src_reg, and the result will be stored in dest_reg.
// work_reg is an intermediate register used for calculations.
inline void _calculate_sigmoid_regs_(const std::uint32_t src_reg, const std::uint32_t work_reg, const std::uint32_t dest_reg)
{
    // ALthough SFPMUL/SFPADD are 2 cycle instructions, we don't need a TTI_NOP
    // because the hardware implicitly stalls if the next instruction depends on results
    TTI_SFPMOV(src_reg, work_reg, 1);                                      // Copy src_reg to work_reg and invert sign bit (take negative of input)
    TTI_SFPNONLINEAR(work_reg, dest_reg, p_sfpnonlinear::EXP_MODE);        // Read value from work_reg, take exponential, load back into dest_reg
    TTI_SFPADD(p_sfpu::LCONST_1, dest_reg, p_sfpu::LCONST_1, work_reg, 0); // Add 1 to dest_reg, store in work_reg, takes 2 cycles
    TTI_SFPNONLINEAR(work_reg, dest_reg, p_sfpnonlinear::RECIP_MODE);      // Read value from work_reg, approximate recip, load back into dest_reg
}

// Calculates SIGMOID for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_sigmoid_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        _calculate_sigmoid_regs_(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG0); // calculate sigmoid using lreg[0] as src and dest, and lreg[1] as work register
        TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS); // store from lreg[0] into dest register
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
