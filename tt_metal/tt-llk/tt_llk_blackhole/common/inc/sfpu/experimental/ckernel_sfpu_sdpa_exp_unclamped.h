// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_sfpu_exp.h"

namespace ckernel
{
namespace sfpu
{

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_lower_clamp_only_(sfpi::vFloat val)
{
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f);

    // Lower clamp only (xlog2 >= 0). Upper clamp is dead when val <= 0 (see file header).
    // One SFPSWAP via sfpi::max, unlike sfpi::vec_min_max's swap through a second operand.
    xlog2 = sfpi::max(xlog2, 0.0f);

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);
    sfpi::vInt fractional_part  = sfpi::exman(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);
    frac              = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en)
    {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::NearestEven);
    }

    return y;
}

template <bool SCALE_EN, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _ckernel_sfpu_exp_accurate_upper_unclamped_(sfpi::vFloat val, const std::uint32_t exp_base_scale_factor)
{
    static_assert(!is_fp32_dest_acc_en, "upper-unclamped exp variant implemented for bf16 dest only");
    if constexpr (SCALE_EN)
    {
        val = val * sfpi::sFloat16b(exp_base_scale_factor);
    }
    return _sfpu_exp_21f_bf16_lower_clamp_only_<is_fp32_dest_acc_en>(val);
}

} // namespace sfpu
} // namespace ckernel
