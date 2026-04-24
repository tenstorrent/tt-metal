// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{
// Calculates SILU for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_silu_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        // Calculate sigmoid using lreg[0] as src, lreg[1] as work register, and lreg[2] as dest (since we need the original value for the final multiply)
        _calculate_sigmoid_regs_(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);         // Multiply lreg[0] * lreg[2], store result in lreg[1]
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS); // store from lreg[1] into dest register
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
