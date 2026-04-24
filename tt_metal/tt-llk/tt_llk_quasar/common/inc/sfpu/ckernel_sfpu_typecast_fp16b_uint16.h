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
inline void _calculate_typecast_fp16b_to_uint16_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP16B, ADDR_MOD_7, 0, dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        TTI_SFPENCC(0, 1);                                                                                      // CC_en <= 1, CC_res <= 1
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0);                                                                      // CC_res <= (LREG0 < 0)
        TTI_SFPLOADI(p_sfpu::LREG0, 0, 0);                                                                      // loads zeros where lreg[0] is negative
        TTI_SFPENCC(0, 1);                                                                                      // CC_en <= 1, CC_res <= 1 (for all)
        TTI_SFP_STOCH_RND(
            p_sfpu::sfp_stochrnd_rnd_mod::NearEven, 0, 0, p_sfpu::LREG0, p_sfpu::LREG1, (1 << 3) | ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_UINT16);
        TT_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::UINT16, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS); // Store from lreg[1] into dest register
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
