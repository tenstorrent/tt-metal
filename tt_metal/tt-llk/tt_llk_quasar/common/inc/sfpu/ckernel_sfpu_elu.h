// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp_21f.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// Vectorized ELU implementation, ported from BH.
// ELU(x) = x                          if x >= 0
// ELU(x) = slope * (exp(x) - 1)       if x <  0
//
// `iterations` is a runtime parameter on Quasar (matches the
// `_calculate_*_(const int iterations, ...)` convention used by the
// existing Quasar SFPU kernels and `_llk_math_eltwise_unary_sfpu_params_`).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void _calculate_elu_(const int iterations, std::uint32_t slope)
{
    sfpi::vFloat s = Converter::as_float(slope);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if (v < 0.0f)
        {
            // is_fp32_dest_acc_en passed as `true` to _sfpu_exp_21f_bf16_ so the
            // exp result keeps fp32 precision; rounding (if needed) is applied
            // once at the end of the operation below.
            sfpi::vFloat v_exp  = _sfpu_exp_21f_bf16_<true>(v) - sfpi::vConst1;
            sfpi::vFloat result = s * v_exp;
            if constexpr (!is_fp32_dest_acc_en)
            {
                result = sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven);
            }
            sfpi::dst_reg[0] = result;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
