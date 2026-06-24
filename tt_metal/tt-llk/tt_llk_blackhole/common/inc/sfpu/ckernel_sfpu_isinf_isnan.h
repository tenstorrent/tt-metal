// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/* Checks if the exponent is 128 and mantissa is 0.
If both conditions are met, the number is marked as
infinity, so '1' is written in the location of the DEST
where the number was stored. Otherwise, `0` is written instead
of the number.
*/
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_isinf_(const sfpi::vFloat& in)
{
    // SFPU microcode
    sfpi::vInt exp   = sfpi::exexp(in);
    sfpi::vInt man   = sfpi::exman(in);
    sfpi::vFloat out = 0.0f;
    v_if (exp == 128 && man == 0)
    {
        out = 1.0f;
    }
    v_endif;
    return out;
}

/* Checks if the sign bit of the floating point number in DEST
is positive. Checks if the exponent is 128 and mantissa is 0.
If all of the three conditions are met, the number is marked as
positive infinity, so '1' is written in the location of the DEST
where the number was stored. Otherwise, `0` is written instead
of the number.
*/
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_isposinf_(const sfpi::vFloat& in)
{
    // SFPU microcode
    sfpi::vInt exp     = sfpi::exexp(in);
    sfpi::vInt man     = sfpi::exman(in);
    sfpi::vInt pos     = sfpi::lz(sfpi::as<sfpi::vUInt>(in));
    sfpi::vFloat out   = 0.0f;
    v_if (pos != 0 && exp == 128 && man == 0)
    {
        out = 1.0f;
    }
    v_endif;
    return out;
}

/* Checks if the sign bit of the floating point number in DEST
is negative. Checks if the exponent is 128 and mantissa is 0.
If all of the three conditions are met, the number is marked as
negative infinity, so '1' is written in the location of the DEST
where the number was stored. Otherwise, `0` is written instead
of the number.
*/
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_isneginf_(const sfpi::vFloat& in)
{
    // SFPU microcode
    sfpi::vInt exp     = sfpi::exexp(in);
    sfpi::vInt man     = sfpi::exman(in);
    sfpi::vInt pos     = sfpi::lz(sfpi::as<sfpi::vUInt>(in));
    sfpi::vFloat out   = 0.0f;
    v_if (pos == 0 && exp == 128 && man == 0)
    {
        out = 1.0f;
    }
    v_endif;
    return out;
}

/* Checks if the exponent is 128 and mantissa is not 0.
If both conditions are met, the number is marked as
nan, so '1' is written in the location of the DEST
where the number was stored. Otherwise, `0` is written instead
of the number.
*/
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_isnan_(const sfpi::vFloat& in)
{
    // SFPU microcode
    sfpi::vInt exp   = sfpi::exexp(in);
    sfpi::vInt man   = sfpi::exman(in);
    sfpi::vFloat out = 0.0f;
    v_if (exp == 128 && man != 0)
    {
        out = 1.0f;
    }
    v_endif;
    return out;
}

template <bool APPROXIMATION_MODE>
inline sfpi::vFloat _calculate_isfinite_(const sfpi::vFloat& v)
{
    // SFPU microcode
    // A number is finite if it's neither infinity nor NaN
    sfpi::vInt exp      = sfpi::exexp(v);
    sfpi::vFloat result = 1.0f; // Assume finite by default

    // If exponent is 128, the number is either infinity or NaN (not finite)
    v_if (exp == 128)
    {
        result = 0.0f;
    }
    v_endif;

    return result;
}

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sfpu_isinf_isnan_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        if constexpr (operation == SfpuType::isinf)
        {
            val = _calculate_isinf_<APPROXIMATION_MODE>(val);
        }
        else if constexpr (operation == SfpuType::isposinf)
        {
            val = _calculate_isposinf_<APPROXIMATION_MODE>(val);
        }
        else if constexpr (operation == SfpuType::isneginf)
        {
            val = _calculate_isneginf_<APPROXIMATION_MODE>(val);
        }
        else if constexpr (operation == SfpuType::isnan)
        {
            val = _calculate_isnan_<APPROXIMATION_MODE>(val);
        }
        else if constexpr (operation == SfpuType::isfinite)
        {
            val = _calculate_isfinite_<APPROXIMATION_MODE>(val);
        }

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
