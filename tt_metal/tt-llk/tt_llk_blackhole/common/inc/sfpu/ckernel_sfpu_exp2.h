// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

sfpi_inline sfpi::vFloat _sfpu_exp2_core_fp32_(sfpi::vFloat x)
{
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(x, k_int);
    sfpi::vFloat f = x - k;

    sfpi::vFloat p = PolynomialEvaluator::eval(
        f,
        sfpi::vConst1,
        0.6931471805599453f,
        0.2402265069591007f,
        0.05550410866482158f,
        0.009618129107628477f,
        0.0013333558146428442f,
        0.0001540353039338164f,
        0.00001525273328683212f);

    sfpi::vInt p_exp = sfpi::exexp(p, sfpi::ExponentMode::NoDebias);
    return sfpi::setexp(p, p_exp + k_int);
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp2_core_bf16_(sfpi::vFloat x)
{
    sfpi::vFloat xlog2 = x + 127.f;

    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);
    sfpi::vInt fractional_part = sfpi::exman(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en)
    {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::NearestEven);
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_exp2_special_or_fp32_(sfpi::vFloat x)
{
    sfpi::vFloat result = sfpi::vConst0;

    sfpi::vInt bits = sfpi::reinterpret<sfpi::vInt>(x);
    sfpi::vInt abs_bits = bits & 0x7fffffff;

    v_if (abs_bits == 0x7f800000)
    {
        v_if (bits < 0)
        {
            result = sfpi::vConst0;
        }
        v_else
        {
            result = std::numeric_limits<float>::infinity();
        }
        v_endif;
    }
    v_elseif (abs_bits > 0x7f800000)
    {
        result = x;
    }
    v_elseif (x >= 128.0f)
    {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif (x <= -127.0f)
    {
        result = sfpi::vConst0;
    }
    v_else
    {
        result = _sfpu_exp2_core_fp32_(x);
    }
    v_endif;

    return result;
}

sfpi_inline sfpi::vFloat _sfpu_exp2_special_or_bf16_(sfpi::vFloat x)
{
    sfpi::vFloat result = sfpi::vConst0;

    sfpi::vInt bits = sfpi::reinterpret<sfpi::vInt>(x);
    sfpi::vInt abs_bits = bits & 0x7fffffff;

    v_if (abs_bits == 0x7f800000)
    {
        v_if (bits < 0)
        {
            result = sfpi::vConst0;
        }
        v_else
        {
            result = std::numeric_limits<float>::infinity();
        }
        v_endif;
    }
    v_elseif (abs_bits > 0x7f800000)
    {
        result = x;
    }
    v_elseif (x >= 128.0f)
    {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif (x <= -127.0f)
    {
        result = sfpi::vConst0;
    }
    v_else
    {
        result = _sfpu_exp2_core_bf16_<true>(x);
    }
    v_endif;

    return sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
}

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];

        if constexpr (is_fp32_dest_acc_en)
        {
            sfpi::dst_reg[0] = _sfpu_exp2_special_or_fp32_(v);
        }
        else
        {
            sfpi::dst_reg[0] = _sfpu_exp2_special_or_bf16_(v);
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
}

} // namespace ckernel::sfpu
