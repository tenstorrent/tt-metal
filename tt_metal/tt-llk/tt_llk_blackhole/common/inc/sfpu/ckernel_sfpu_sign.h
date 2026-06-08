// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_is_fp16_zero.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sign_(const int iterations, std::uint32_t)
{
// All params are in FP16 format
// uint format = 1;
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::vFloat res = 1.0f;
        v_if (v < 0.0F)
        {
            res = -1.0f;
        }
        v_elseif (_sfpu_is_fp16_zero_(v))
        {
            res = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
