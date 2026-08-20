// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

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
    // Use 1.44 as first guess at x, ideal value would be 1.33.
    // Grayskull has hardwired 1.44 and uses it to avoid a load.
    // We use it here for consistency.
    sfpi::vFloat vConstLn2Recip = 1.442695f;
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
    sfpi::vFloat out = setexp(result, new_exp);

    // The pole. in == 0 makes setexp(val, 126) discard the magnitude, so the exponent
    // difference above lands on 126 - exexp(0) = 254 -- an ordinary finite, measured as
    // 1.7e38 -- where infinity needs 255. The v_if(new_exp < 0) block above guards only the
    // opposite, underflow end. Issue #52930 finding 4.
    //
    // Applied after the setexp rather than alongside the underflow guard: writing an infinity
    // into `result` first and then running setexp over it would overwrite the exponent field
    // that makes it an infinity.
    //
    // Compared on setsgn(in, 0) rather than a bare in == 0.0F because SFPSETCC's contract
    // excludes negative zero (VectorUnit.md); measured, the bare compare does not fire for
    // -0.0 and leaves that pole at 1.7e38. Clearing the sign first costs ~2 percentage points
    // of the total, and it is what makes -0.0 reach this guard at all. The result there is
    // then IEEE's: measured 1/-0.0 = -inf on the unpack-to-dest pipelines, where a real -0.0
    // survives to the LREG and the caller-side v_if(in < 0.0) re-signs the magnitude. On the
    // pipelines that flush -0.0 to +0.0 before the kernel sees it the answer is +inf, which
    // is the flush showing through rather than this guard.
    v_if (sfpi::setsgn(in, 0) == 0.0F)
    {
        out = std::numeric_limits<float>::infinity();
    }
    v_endif;
    return out;
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
