// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_abs_(const int iterations)
{
    // SFPU microcode
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::abs(v);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
