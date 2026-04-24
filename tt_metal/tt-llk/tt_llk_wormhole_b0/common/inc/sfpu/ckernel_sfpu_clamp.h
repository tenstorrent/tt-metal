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
inline void _calculate_clamp_(
    std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    // All params are in FP16 format
    // param0 = min
    // param1 = max

    // uint format = (param0 >> 16)&0x1;

    // SFPU microcode
    sfpi::vFloat min    = sfpi::sFloat16a(param0);
    sfpi::vFloat max    = sfpi::sFloat16a(param1);
    sfpi::vFloat offset = sfpi::sFloat16b(param2); // 12 bits
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];

        v_if (val < min)
        {
            val = min;
        }
        v_elseif (val >= max)
        {
            val = max;
        }
        v_endif;

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = val + offset;

        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
