// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
        sfpi::dst_reg[0].mode<sfpi::SFPSTORE_MOD0_FMT_FP16A>() = sfpi::float_to_fp16a(sfpi::dst_reg[0], sfpi::RoundMode::NearestEven);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
