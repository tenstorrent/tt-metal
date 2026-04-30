// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_expm1_cw.h"

namespace ckernel::sfpu
{

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_selu_(std::uint32_t scale, std::uint32_t alpha)
{
    const sfpi::vFloat scale_val   = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
// unroll 2: with expm1_cw_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x      = sfpi::dst_reg[0];
        sfpi::vFloat result = scale_alpha * expm1_cw_clamped(x);

        v_if (x >= 0.0f)
        {
            result = scale_val * x;
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
