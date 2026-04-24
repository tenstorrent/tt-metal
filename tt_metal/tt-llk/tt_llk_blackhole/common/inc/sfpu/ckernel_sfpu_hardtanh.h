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
inline void _calculate_hardtanh_(
    std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat p0                           = sfpi::sFloat16b(param0);
    sfpi::vFloat p1                           = sfpi::sFloat16b(param1);
    sfpi::vFloat p2                           = sfpi::sFloat16b(param2);
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];

        val += p0; // 12 bits
        v_if (val < 0.0f)
        {
            val = 0.0f;
        }
        v_endif;

        val += p1; // 12 bits
        v_if (val >= 0.0f)
        {
            val = 0.0f;
        }
        v_endif;

        val += p2; // 12 bits

        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = val;

        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
