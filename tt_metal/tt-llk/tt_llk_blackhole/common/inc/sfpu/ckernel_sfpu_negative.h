// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_(std::uint32_t dst_index_in, std::uint32_t dst_index_out)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val                                 = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = -val;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_int_(std::uint32_t dst_index_in, std::uint32_t dst_index_out)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vInt val                                   = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = -val;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
