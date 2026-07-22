// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Element-wise clamp out = min(max(x, min_val), max_val); param0/param1 are FP16A bounds (mirrors the
// Blackhole clamp, minus the offset). Uses v_if compares (sfpi::min/max's SFPSWAP is sign-magnitude and
// mis-orders negatives) and relies on vConstNeg1/LREG11 == -1.0, re-established per launch by
// _init_sfpu_config_reg_.
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_clamp_(std::uint32_t param0, std::uint32_t param1)
{
    sfpi::vFloat min_val = sfpi::sFloat16a(param0);
    sfpi::vFloat max_val = sfpi::sFloat16a(param1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if (val < min_val)
        {
            val = min_val;
        }
        v_elseif (val >= max_val)
        {
            val = max_val;
        }
        v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
