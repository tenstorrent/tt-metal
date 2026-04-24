// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
// Calculates RECIP for number of rows of output SFPU ops (Quasar = 2 rows)
template <bool APPROXIMATION_MODE>
inline void _calculate_reciprocal_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(
            p_sfpu::LREG0,
            p_sfpu::sfpmem::DEFAULT,
            ADDR_MOD_7,
            0,
            dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

        // SFPARECIP, approx version of reciprocal
        if constexpr (APPROXIMATION_MODE)
        {
            TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::RECIP_MODE); // Read value from lreg[0], approximate recip, load back into lreg[1]
        }

        // Store from lreg[1] into dest register
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS);
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
