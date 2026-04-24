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
// Calculates Typecast for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_typecast_int32_to_fp32_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        // TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG2, 3); //convert from 2s complement to sign+magnitude
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG1, 0); // convert from int32 sign+mag to fp32 using rnd nearest even
        TT_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS); // Store from lreg[1] into dest register
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
