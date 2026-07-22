// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

/*
 * The _sfpu_exp_fp32_accurate_ code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2021, Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// Round-to-nearest-even of a float to its integer value, returning both the rounded float (result)
// and the integer (k_int). Uses the Hacker's Delight 2^23 + 2^22 trick: adding that constant forces
// the fractional bits out, and differencing the raw bit patterns recovers the integer. Only uses
// add/sub plus a bit reinterpret, so it is portable to Quasar (unlike the sign-magnitude round the
// Blackhole kernel interleaves). Valid for |z| < 2^22, which covers exp's reduced argument.
sfpi_inline sfpi::vFloat _sfpu_round_to_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int)
{
    const sfpi::vFloat c231 = 12582912.0f; // 2^23 + 2^22
    sfpi::vFloat tmp        = z + c231;
    k_int                   = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);
    return tmp - c231;
}

sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_(sfpi::vFloat a)
{
    sfpi::vInt i;
    sfpi::vFloat f, r, j;

    // j = round(a / ln2) (as a float) and i = the same value as an integer, interleaved with the
    // first coefficient of the polynomial.
    r = 1.37805939e-3f;
    j = _sfpu_round_to_nearest_int32_(1.442695f * a, i);

    // f = a - j*ln2 (two-part Cody-Waite).
    f = j * -6.93145752e-1f + a;
    f = j * -1.42860677e-6f + f;

    // r = exp(f) on [-ln2/2, ln2/2] via a degree-6 minimax polynomial in Horner form.
    r = r * f + 8.37312452e-3f; // 0x1.125edcp-7
    r = r * f + 4.16695364e-2f; // 0x1.555b5ap-5
    r = r * f + 1.66664720e-1f; // 0x1.555450p-3
    r = r * f + 4.99999851e-1f; // 0x1.fffff6p-2
    r = r * f + 1.0f;
    r = r * f + 1.0f;

    // exp(a) = 2^i * exp(f). Construct 2^i by writing the biased exponent (i + 127) directly into
    // the IEEE-754 exponent field. This is equivalent to sfpi::setexp on top of a 1.0 seed (setexp
    // and exexp are correct on Quasar); the int add / shift-left / reinterpret used here are simply
    // cheaper and keep the whole path in fp32 / two's-complement. Correct across fp32 exp's
    // representable domain (|a| < ~88); larger-magnitude inputs are the LUT/approx path's concern.
    //
    // NB: the Quasar port bug was NOT here. It was the Blackhole kernel's sign-magnitude rounding
    // (abs(as<vInt>(convert<vSMag16>(x))) + copysgn feeding a two's-complement add), which relies on
    // Blackhole's integer-format behaviour. _sfpu_round_to_nearest_int32_ above replaces that and is
    // the actual fix.
    sfpi::vFloat two_i = sfpi::as<sfpi::vFloat>((i + 127) << 23);
    return r * two_i;
}

// Calculates EXP over a full tile. Quasar exposes exactly two implementations:
//   - approximate exp via the HW nonlinear lookup table (sfpi::approx_exp), and
//   - full-precision fp32 exp (_sfpu_exp_fp32_accurate_, ported from Blackhole).
// The LUT is ~1 ULP once the result lands in a bf16 Dest, so the accurate path is only worth
// running for a 32-bit Dest in non-approximate mode; every bf16 case (and any explicit approx
// request) uses the LUT.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_exp_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // load x from dest (SFPLOAD)

        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en || APPROXIMATION_MODE)
        {
            result = sfpi::approx_exp(val);
        }
        else
        {
            result = _sfpu_exp_fp32_accurate_(val);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
