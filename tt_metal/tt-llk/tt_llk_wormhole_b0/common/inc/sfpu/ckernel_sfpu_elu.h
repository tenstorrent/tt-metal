// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_expm1_cw.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_elu_(std::uint32_t slope)
{
    sfpi::vFloat alpha = Converter::as_float(slope);
// unroll 2: with expm1_cw_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x      = sfpi::dst_reg[0];
        sfpi::vFloat result = alpha * expm1_cw_clamped(x);

        v_if (x >= 0.0f)
        {
            result = x;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en)
        {
            result = sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
