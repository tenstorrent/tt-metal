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
// Implements leaky relu which return x when x > 0 and x*slope when x < 0
inline void _calculate_lrelu_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations, const std::uint32_t slope)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    TTI_SFPLOADI(p_sfpu::LREG2, 0 /*Float16_b*/, (slope >> 16)); // store slope in LREG2
    TTI_SFPENCC(1, 2);                                           // enable cc
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0);                                            // condition - if value in LREG0 is negative //will set cc result reg
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Multiply and add - LREG0 * LREG2 + LCONST_0 (x * slope + 0)
        TTI_SFPENCC(0, 0);                                                            // clear cc result reg
        TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS); // store from lreg[0] into dest register
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
    TTI_SFPENCC(0, 2); // disable cc
}

// Implements relu min which returns x when x > threshold, otherwise return 0
inline void _calculate_relu_min_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations, const std::uint32_t threshold)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    TTI_SFPLOADI(p_sfpu::LREG2, 0 /*Float16_b*/, (threshold >> 16)); // store threshold in LREG2
    TTI_SFPENCC(1, 2);                                               // enable cc
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        // where lreg0 < threshold, load 0 into LREG0[x]
        TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LREG2, 0x1 /*cc mode*/);
        TTI_SFPLOADI(p_sfpu::LREG0, 0, 0);                                                    // Load 0 into lreg0[x]
        TTI_SFPENCC(0, 0);                                                                    // clear cc result reg
        TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS); // store from lreg[0] into dest register
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
    TTI_SFPENCC(0, 2); // disable cc
}

// Implements relu max
inline void _calculate_relu_max_(std::uint32_t dst_tile_index_in, std::uint32_t dst_tile_index_out, const int iterations, const std::uint32_t threshold)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    TTI_SFPLOADI(p_sfpu::LREG2, 0 /*Float16_b*/, (threshold >> 16)); // store threshold in LREG2
    TTI_SFPENCC(1, 2);                                               // enable cc
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, dst_tile_index_in * SFP_DST_TILE_ROWS); // load from dest into lreg[0]
        // If x > threshold, x = threshold
        TTI_SFPGT(0, p_sfpu::LREG2, p_sfpu::LREG0, 0x1 /*cc mode*/);
        TTI_SFPMOV(p_sfpu::LREG2 /*src*/, p_sfpu::LREG0 /*dest*/, 0); // copy threshold value into LREG0 where LREG0 > threshold
        // Classic Relu: Max(x, 0)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0);                                                    // condition - if value in LREG0 is negative
        TTI_SFPLOADI(p_sfpu::LREG0, 0, 0);                                                    // Load 0 into lreg0[x]
        TTI_SFPENCC(0, 0);                                                                    // reset cc
        TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, dst_tile_index_out * SFP_DST_TILE_ROWS); // store from lreg[0] into dest register
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
    TTI_SFPENCC(0, 2); // disable cc
}

} // namespace sfpu
} // namespace ckernel
