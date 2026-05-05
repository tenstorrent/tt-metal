// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _cast_fp32_to_fp16a_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        // sfpi::vFloat val = sfpi::dst_reg[0];
        // sfpi::dst_reg[0] = sfpi::float_to_fp16a(val, sfpi::RoundMode::NearestEven);
        TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 8);
        TTI_SFPSTORE(0, 1, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
