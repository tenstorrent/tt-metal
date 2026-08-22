// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int RECIPROCAL_ITERATIONS>
sfpi_inline sfpi::vFloat _sqrt_compat_(sfpi::vFloat val)
{
    sfpi::vFloat result;
    if constexpr (APPROXIMATION_MODE)
    {
        sfpi::vUInt magic = (127 << 7) << 16;

        // sqrt initial approximation
        //  adjust bias
        sfpi::vUInt val_s = magic + sfpi::as<sfpi::vUInt>(val);

        // approximation of square root
        val_s >>= 1;
        result = sfpi::as<sfpi::vFloat>(val_s);
    }
    else
    {
        // Recip root method
        //// Init approx
        // u.i = SQRT_MAGIC_F - (u.i >> 1);
        v_if (val != 0.0f)
        {
            sfpi::vUInt magic   = 0x5f37 << 16;
            sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));

            // Reciproot iterations
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                // x*r*(1.5f - xhalf*r*r);
                approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
            }

            result = approx * val;
        }
        v_else
        {
            result = val;
        }
        v_endif;
    }
    return result;
}

template <int max_iter = 3>
sfpi_inline sfpi::vFloat _reciprocal_compat_(const sfpi::vFloat in)
{
    // Force sign to 1 (make number negative)
    sfpi::vFloat val = sfpi::setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Newton seed for 1/m on the reduced domain m in [0.5, 1): the minimax-optimal constant
    // seed is 4/3 (error at m=0.5 and m=1 balance at +/-1/3, vs +/-0.44 for 1.442695).
    // 0x1.554p0f is the nearest fp16a-representable value to 4/3, so it loads in a single
    // SFPLOADI (fp16a immediate) instead of the two SFPLOADI needed for 1.442695, while
    // keeping fp32 accuracy after 3 iterations at ~1.53e-04 max relative error vs 1.475e-03.
    sfpi::vFloat vConstLn2Recip = 0x1.554p0f;
    sfpi::vFloat two            = 2.0f;
    sfpi::vFloat result         = vConstLn2Recip * (val * vConstLn2Recip + two);

    for (int s_iter = 0; s_iter < (max_iter - 1); s_iter++)
    {
        result = result * (val * result + two);
    }

    sfpi::vInt orig_exp = exexp(in);
    sfpi::vInt new_exp  = exexp(result);

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0)
    {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result  = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return setexp(result, new_exp);
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en>
inline void _calculate_rsqrt_compat_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::dst_reg[0] = _sqrt_compat_<APPROXIMATION_MODE, 2>(sfpi::dst_reg[0]);
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat out = _reciprocal_compat_<APPROXIMATION_MODE ? 2 : 3>(in);
        v_if (in < 0.0)
        {
            out = -out;
        }
        v_endif;
        if constexpr (!(fp32_dest_acc_en || APPROXIMATION_MODE))
        {
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en>
inline void _calculate_sqrt_compat_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::dst_reg[0] = _sqrt_compat_<APPROXIMATION_MODE, 2>(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en>
inline void _calculate_reciprocal_compat_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in  = sfpi::dst_reg[0];
        sfpi::vFloat out = _reciprocal_compat_<APPROXIMATION_MODE ? 2 : 3>(in);
        v_if (in < 0.0)
        {
            out = -out;
        }
        v_endif;
        if constexpr (!(fp32_dest_acc_en || APPROXIMATION_MODE))
        {
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
