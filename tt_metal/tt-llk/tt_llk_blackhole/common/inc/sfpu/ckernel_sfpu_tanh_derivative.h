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

template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH, int ITERATIONS>
inline void _calculate_tanh_derivative_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations)
{
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vUInt l0                            = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1                            = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2                            = sfpi::l_reg[sfpi::LRegs::LReg2];

    // tanh'(x) = 1 - (tanh(x))^2
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];

        if constexpr (!WITH_PRECOMPUTED_TANH)
        {
            val = lut(val, l0, l1, l2);
        }

        val                                              = val * (-val) + sfpi::vConst1;
        sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = val;

        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
}

} // namespace sfpu
} // namespace ckernel
