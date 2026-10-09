// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
// Calculates Typecast for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_typecast_uint16_to_fp32_rows()
{
    const sfpi::vUInt16 value = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();

    sfpi::dst_reg[0].mode<sfpi::DataLayout::F32>() = sfpi::convert<sfpi::vFloat>(value, sfpi::RoundMode::NearestEven);
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_typecast_uint16_to_fp32_rows();
        sfpi::dst_reg++; // increments by 2 rows (SFP_DESTREG_STRIDE), one SFPU pass
    }
}

} // namespace sfpu
} // namespace ckernel
