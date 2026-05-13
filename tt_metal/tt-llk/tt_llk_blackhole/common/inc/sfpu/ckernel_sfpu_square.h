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
inline void _calculate_square_(std::uint32_t dst_index_in, std::uint32_t dst_index_out)
{
    // SFPI-style: addressing via sfpi::dst_reg[] keeps the read row and the write
    // row in sync as we advance, which is required when dst_index_in != dst_index_out.
    // Earlier TT_SFP*-based implementations only kept addressing correct for the
    // in == out case (offset 0); see ckernel_sfpu_abs.h for the same pattern.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v                                     = sfpi::dst_reg[0];
        sfpi::dst_reg[(dst_index_out - dst_index_in) * 32] = v * v;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
