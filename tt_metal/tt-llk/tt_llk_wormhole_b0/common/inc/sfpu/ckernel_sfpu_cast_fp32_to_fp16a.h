// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _cast_fp32_to_fp16a_(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 8);
        TT_SFPSTORE(0, 1, 3, (dst_index_out - dst_index_in) * TILE_R_DIM);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
