// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <climits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// #41922: CONSTRUCTION ZONE
// Implemented:Standard calling API, old bodies
// ToDo: use sfpi API

// compute truncate to zero
sfpi_inline sfpi::vFloat _trunc_body_(sfpi::vFloat val)
{
    sfpi::vFloat result = val;
    sfpi::vInt exp      = sfpi::exexp(val);
    v_if (exp < 0)
    {
        const sfpi::vFloat zero = 0.0f;
        result                  = sfpi::copysgn(zero, val);
    }
    v_elseif (exp < 23)
    {
        const sfpi::vInt shift = exp - 23;
        result                 = sfpi::as<sfpi::vFloat>(sfpi::shft(sfpi::shft(sfpi::as<sfpi::vUInt>(val), shift), -shift));
    }
    v_else
    {
        result = val;
    }
    v_endif;
    return result;
}

// compute floor
sfpi_inline sfpi::vFloat _floor_body_(sfpi::vFloat val)
{
    sfpi::vFloat result = _trunc_body_(val);
    v_if (val < result)
    {
        result -= 1.0f;
    }
    v_endif;
    return result;
}

// computes ceil
sfpi_inline sfpi::vFloat _ceil_body_(sfpi::vFloat val)
{
    sfpi::vFloat result = _trunc_body_(val);
    v_if (val > result)
    {
        result += 1.0f;
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
        sfpi::vFloat x   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = x - _trunc_body_(x);
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _round_even_(sfpi::vFloat v)
{
    // Create a temporary copy tmp = abs(v).
    sfpi::vFloat tmp = sfpi::setsgn(v, 0);
    // For all 0 ≤ x < 2**23, x + 2**23 will shift out the fractional part with round-to-nearest-even.
    tmp += 0x1.p23f;
    // Hide SFPNOP; extract exponent.
    sfpi::vInt exp = sfpi::exexp(v);
    // Subtract 2**23 to restore exponent.
    tmp += -0x1.p23f;
    // Hide SFPNOP; check exponent.  If x ≥ 2**23, then there is no fractional part.
    v_if (exp < 23)
    {
        // v.{Exp,Man}=tmp.{Exp,Man}; retaining original sign.
        v = sfpi::copysgn(tmp, v);
    }
    v_endif;
    return v;
}

template <bool APPROXIMATE, int ITERATIONS = 8>
void _calculate_round_(const int decimals)
{
    const auto exp10i = [](int n)
    {
        if (n > 38) // 38 is max decimal places float32 can store for positive values
        {
            return 1.0F / 0.0F;
        }

        if (n < -45) // 45 is max decimal places float32 can store for negative values
        {
            return 0.0F;
        }

        return PRECOMPUTED_POW10_TABLE[n + 45];
    };

    const sfpi::vFloat coeff   = exp10i(decimals);
    const sfpi::vFloat inverse = exp10i(-decimals);

    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v      = sfpi::dst_reg[0];
        sfpi::vFloat result = inverse * _round_even_(v * coeff);
        sfpi::dst_reg[0]    = result;
        sfpi::dst_reg++;
    }
}

// Performs stochastic rounding of values in DST from fp32 to fp16b format.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
sfpi_inline void _calculate_stochastic_round_()
{
#pragma GCC unroll ITERATIONS
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat x   = sfpi::dst_reg[0];
        x                = sfpi::convert<sfpi::vFloat16b>(x, sfpi::RoundMode::NearestStochastic);
        sfpi::dst_reg[0] = x;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
