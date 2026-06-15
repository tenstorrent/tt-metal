// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const std::uint32_t log_base_scale_factor, const std::uint32_t dst_idx = 0)
{
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat x  = setexp(in, 127); // set exp to exp bias (put in range of 1-2)

    // XXXXXX ask Namal? if we can derive the coefficients below to higher precision
    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D'):
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    // XXXXX try variants of the below: B'=.7122, C'=2.0869
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    auto exp = sfpi::convert<sfpi::vSMag>(exexp(in));

    sfpi::vFloat expf      = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result    = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING)
    {
        result *= sfpi::sFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (in == 0.0F)
    { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base)
{
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, 127); // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    auto exp          = sfpi::convert<sfpi::vSMag>(exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat vConstLn2  = 0.692871f;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if (base == 0.0f)
    {
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return log_result;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _calculate_log1p_body_no_init_(sfpi::vFloat a)
{
    sfpi::vFloat u = a + sfpi::vConst1;
    sfpi::vFloat r = std::numeric_limits<float>::quiet_NaN();

    v_if (u >= 0.0f)
    {
        sfpi::vFloat three_quarters = 0.75f;
        sfpi::vInt e = sfpi::reinterpret<sfpi::vInt>(three_quarters);
        sfpi::vFloat e_float;

        e = sfpi::reinterpret<sfpi::vInt>(u) - e;
        e = sfpi::reinterpret<sfpi::vInt>(sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(e), 0));

        sfpi::vFloat m = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - e);
        sfpi::vFloat neg_four = -4.0f;
        sfpi::vFloat s = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(neg_four) - e);
        sfpi::vFloat neg_quarter = -0.25f;
        sfpi::vFloat neg1 = sfpi::vConstNeg1;
        sfpi::vFloat t = __builtin_rvtt_sfpmad(neg_quarter.get(), s.get(), neg1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vMag abs_e = sfpi::abs(e);

        if constexpr (is_fp32_dest_acc_en)
        {
            m = m + t;
            r = -0x1.92cp-5f;
            r = r * m + 0x1.b84p-4f;
            r = r * m + -0x1.0c4p-3f;
            r = r * m + 0x1.274p-3f;
            r = r * m + -0x1.55p-3f;
            r = r * m + 0x1.998p-3f;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::NearestEven);
            r = r * m + -0x1.00001ap-2f;
            s = m * m;
            r = r * m + 0x1.555572p-2f;
            r = r * m + -0.5f;
        }
        else
        {
            m = m + t;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::NearestEven);
            r = neg_quarter * m + 0x1.744p-2f;
            s = m * m;
            r = r * m + -0x1.008p-1f;
        }

        e_float = sfpi::copysgn(e_float, sfpi::reinterpret<sfpi::vFloat>(e));
        r = r * s + m;
        sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
        r = e_float * (0.693147182f * 1.19209290e-7f) + r;

        v_if (sfpi::reinterpret<sfpi::vInt>(u) >= sfpi::reinterpret<sfpi::vInt>(infinity))
        {
            r = u;
        }
        v_endif;
    }
    v_endif;

    return r;
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

template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = 0.692871f; // ln2

    // XXXXX could do these to higher precision
    sfpi::vConstFloatPrgm1 = 0.1058f;
    sfpi::vConstFloatPrgm2 = -0.7166f;
}

} // namespace sfpu
} // namespace ckernel
