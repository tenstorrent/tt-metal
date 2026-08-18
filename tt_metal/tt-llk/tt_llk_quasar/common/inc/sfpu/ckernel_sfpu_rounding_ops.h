// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ckernel_sfpu_compat.h"
#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// The Quasar implementation deliberately uses the SFPI conversion API instead
// of Blackhole's TTI_SFPLOADI/SFPEXEXP/SFPSHFT2 sequence.  Quasar does not offer
// a vFloat -> vSMag round-toward-zero cast, so round-to-nearest-even is corrected
// by one using only the two known-good vFloat comparison operators.  Values at
// or beyond 2^23 are already integral in FP32 and bypass the conversion; this
// also preserves infinities and NaNs.
sfpi_inline sfpi::vFloat _trunc_body_(sfpi::vFloat value)
{
    sfpi::vFloat result = value;
    v_if (sfpi::abs(value) <= 0x1.0p23f)
    {
        const sfpi::vSMag integral = sfpi::convert<sfpi::vSMag>(value, sfpi::RoundMode::NearestEven);
        sfpi::vFloat rounded       = sfpi::convert<sfpi::vFloat>(integral, sfpi::RoundMode::Nearest);
        v_if (value > 0.0f)
        {
            v_if (rounded > value)
            {
                rounded = rounded - 1.0f;
            }
            v_endif;
        }
        v_elseif (sfpi::vFloat(0.0f) > value)
        {
            v_if (value > rounded)
            {
                rounded = rounded + 1.0f;
            }
            v_endif;
        }
        v_endif;
        result = rounded;
    }
    v_endif;
    return result;
}

sfpi_inline sfpi::vFloat _floor_body_(sfpi::vFloat value)
{
    const sfpi::vFloat truncated = _trunc_body_(value);
    sfpi::vFloat result          = truncated;
    v_if (truncated > value)
    {
        result = truncated - 1.0f;
    }
    v_endif;
    return result;
}

sfpi_inline sfpi::vFloat _ceil_body_(sfpi::vFloat value)
{
    const sfpi::vFloat truncated = _trunc_body_(value);
    sfpi::vFloat result          = truncated;
    v_if (value > truncated)
    {
        result = truncated + 1.0f;
    }
    v_endif;
    return result;
}

inline constexpr std::array<float, 84> PRECOMPUTED_POW10_TABLE = {
    1e-45F, 1e-44F, 1e-43F, 1e-42F, 1e-41F, 1e-40F, 1e-39F, 1e-38F, 1e-37F, 1e-36F, 1e-35F, 1e-34F, 1e-33F, 1e-32F, 1e-31F, 1e-30F, 1e-29F,
    1e-28F, 1e-27F, 1e-26F, 1e-25F, 1e-24F, 1e-23F, 1e-22F, 1e-21F, 1e-20F, 1e-19F, 1e-18F, 1e-17F, 1e-16F, 1e-15F, 1e-14F, 1e-13F, 1e-12F,
    1e-11F, 1e-10F, 1e-9F,  1e-8F,  1e-7F,  1e-6F,  1e-5F,  1e-4F,  1e-3F,  1e-2F,  1e-1F,  1e0F,   1e1F,   1e2F,   1e3F,   1e4F,   1e5F,
    1e6F,   1e7F,   1e8F,   1e9F,   1e10F,  1e11F,  1e12F,  1e13F,  1e14F,  1e15F,  1e16F,  1e17F,  1e18F,  1e19F,  1e20F,  1e21F,  1e22F,
    1e23F,  1e24F,  1e25F,  1e26F,  1e27F,  1e28F,  1e29F,  1e30F,  1e31F,  1e32F,  1e33F,  1e34F,  1e35F,  1e36F,  1e37F,  1e38F,
};

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_floor_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = _floor_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_ceil_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = _ceil_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = _trunc_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_frac_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::dst_reg[0]         = value - _trunc_body_(value);
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _round_even_(sfpi::vFloat value)
{
    sfpi::vFloat result = value;
    v_if (sfpi::abs(value) <= 0x1.0p23f)
    {
        const sfpi::vSMag integral = sfpi::convert<sfpi::vSMag>(value, sfpi::RoundMode::NearestEven);
        result                     = sfpi::convert<sfpi::vFloat>(integral, sfpi::RoundMode::Nearest);
    }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_round_(const int decimals)
{
    const auto exp10i = [](int n)
    {
        if (n > 38)
        {
            return 1.0F / 0.0F;
        }
        if (n < -45)
        {
            return 0.0F;
        }
        return PRECOMPUTED_POW10_TABLE[n + 45];
    };

    const sfpi::vFloat coefficient = exp10i(decimals);
    const sfpi::vFloat inverse     = exp10i(-decimals);
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = inverse * _round_even_(sfpi::dst_reg[0] * coefficient);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_stochastic_round_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        const sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::dst_reg[0]         = sfpi::convert<sfpi::vFloat16b>(value, sfpi::RoundMode::NearestStochastic);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
