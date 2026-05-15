// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"
#include "ckernel_sfpu_polyval.h"

namespace ckernel
{
namespace sfpu
{

namespace log_coef
{
    constexpr float LN2      = 0.693147180559945309417f;
    constexpr float INV_LN2  = 1.442695040888963407359f;
    constexpr float C1 = 0.999999990f;
    constexpr float C2 = -0.499999620f;
    constexpr float C3 = 0.333314564f;
    constexpr float C4 = -0.249913470f;
    constexpr float C5 = 0.199739850f;
    constexpr float C1_3 = 0.999473f;
    constexpr float C2_3 = -0.492974f;
    constexpr float C3_3 = 0.327104f;
}

sfpi_inline sfpi::vFloat _calculate_log_body_fp32_(sfpi::vFloat in)
{
    sfpi::vInt exp_raw  = sfpi::exexp(in);
    sfpi::vFloat mantissa = sfpi::setexp(in, 127);
    sfpi::vFloat t = mantissa - 1.0f;

    sfpi::vFloat log_mantissa = PolynomialEvaluator::eval(
        t,
        log_coef::C1, log_coef::C2, log_coef::C3, log_coef::C4, log_coef::C5
    );

    sfpi::vInt exp = sfpi::setsgn(exp_raw, 0);
    sfpi::vFloat expf = sfpi::int32_to_float(exp, sfpi::RoundMode::NearestEven);
    sfpi::vFloat result = expf * log_coef::LN2 + log_mantissa;

    v_if (in == 0.0F)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return result;
}

sfpi_inline sfpi::vFloat _calculate_log_body_bf16_(sfpi::vFloat in)
{
    sfpi::vInt exp_raw   = sfpi::exexp(in);
    sfpi::vFloat mantissa = sfpi::setexp(in, 127);
    sfpi::vFloat t = mantissa - 1.0f;

    sfpi::vFloat log_mantissa = PolynomialEvaluator::eval(
        t,
        log_coef::C1_3, log_coef::C2_3, log_coef::C3_3
    );

    sfpi::vInt exp = sfpi::setsgn(exp_raw, 0);
    sfpi::vFloat expf = sfpi::int32_to_float(exp, sfpi::RoundMode::NearestEven);
    sfpi::vFloat result = expf * log_coef::LN2 + log_mantissa;

    v_if (in == 0.0F)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return result;
}

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const std::uint32_t log_base_scale_factor, const std::uint32_t dst_idx = 0)
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in    = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat x     = sfpi::setexp(in, 127);
    sfpi::vInt exp_raw = sfpi::exexp(in);

    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871f) + -1.4753f;

    sfpi::vInt exp = sfpi::setsgn(exp_raw, 0);
    sfpi::vFloat expf      = sfpi::int32_to_float(exp, sfpi::RoundMode::NearestEven);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result    = expf * vConstLn2 + series_result;

    if constexpr (HAS_BASE_SCALING)
    {
        result *= sfpi::sFloat16a(log_base_scale_factor);
    }

    v_if (in == 0.0F)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base)
{
    sfpi::vFloat x = sfpi::setexp(base, 127);

    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    sfpi::vInt exp_raw = sfpi::exexp(base);
    sfpi::vInt exp     = sfpi::setsgn(exp_raw, 0);
    sfpi::vFloat expf  = sfpi::int32_to_float(exp, sfpi::RoundMode::NearestEven);

    sfpi::vFloat vConstLn2  = 0.692871f;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result;

    v_if (base == 0.0f)
    {
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return log_result;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(const int iterations, std::uint32_t log_base_scale_factor)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
inline void _calculate_log_bf16_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::dst_reg[0] = _calculate_log_body_bf16_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
inline void _calculate_log_fp32_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::dst_reg[0] = _calculate_log_body_fp32_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = 0.692871f;
    sfpi::vConstFloatPrgm1 = 0.1058f;
    sfpi::vConstFloatPrgm2 = -0.7166f;
}

} // namespace sfpu
} // namespace ckernel
