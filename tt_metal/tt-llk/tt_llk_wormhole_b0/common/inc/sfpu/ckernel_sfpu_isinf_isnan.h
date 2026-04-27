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
    sfpi::vInt man   = sfpi::exman9(in);
    sfpi::vFloat out = sfpi::vConst0;
    v_if (exp == 128 && man == 0)
    {
        out = sfpi::vConst1;
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
    sfpi::vInt man     = sfpi::exman9(in);
    sfpi::vFloat out   = sfpi::vConst0;
    sfpi::vInt signbit = sfpi::reinterpret<sfpi::vInt>(in) & 0x80000000; // returns 0 for +ve value
    v_if (signbit == 0 && exp == 128 && man == 0)
    {
        out = sfpi::vConst1;
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
    sfpi::vInt man     = sfpi::exman9(in);
    sfpi::vFloat out   = sfpi::vConst0;
    sfpi::vInt signbit = sfpi::reinterpret<sfpi::vInt>(in) & 0x80000000; // returns 0x80000000 for -ve value
    v_if (signbit == 0x80000000 && exp == 128 && man == 0)
    {
        out = sfpi::vConst1;
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
    sfpi::vInt man   = sfpi::exman9(in);
    sfpi::vFloat out = sfpi::vConst0;
    v_if (exp == 128 && man != 0)
    {
        out = sfpi::vConst1;
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
    sfpi::vFloat result = sfpi::vConst1; // Assume finite (1.0f) by default

    // If exponent is 128, the number is either infinity or NaN (not finite)
    v_if (exp == 128)
    {
        result = sfpi::vConst0;
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
        sfpi::vFloat in = sfpi::dst_reg[0];

        if constexpr (operation == SfpuType::isinf)
        {
            sfpi::dst_reg[0] = _calculate_isinf_<APPROXIMATION_MODE>(in);
        }
        else if constexpr (operation == SfpuType::isposinf)
        {
            sfpi::dst_reg[0] = _calculate_isposinf_<APPROXIMATION_MODE>(in);
        }
        else if constexpr (operation == SfpuType::isneginf)
        {
            sfpi::dst_reg[0] = _calculate_isneginf_<APPROXIMATION_MODE>(in);
        }
        else if constexpr (operation == SfpuType::isnan)
        {
            sfpi::dst_reg[0] = _calculate_isnan_<APPROXIMATION_MODE>(in);
        }
        else if constexpr (operation == SfpuType::isfinite)
        {
            sfpi::dst_reg[0] = _calculate_isfinite_<APPROXIMATION_MODE>(in);
        }

        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
