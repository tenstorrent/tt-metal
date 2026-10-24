// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_polyval.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

// ======================================================================
// Shared helper: exp(x) - 1 via Cody-Waite range reduction + factored
// expm1 polynomial. Used by ELU, CELU and SELU.
//
// Algorithm: x = k*ln(2) + r, |r| <= ln(2)/2
//   expm1(r) = r * h(r), h(r) = minimax polynomial on [-ln2/2, ln2/2]
//   exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
//
// BF16 h degree 4: max abs error = 1.60e-7 (Sollya remez)
// FP32 h degree 5: max abs error = 8.67e-9 (Sollya remez)
// ======================================================================

constexpr float CW_INV_LN2    = 1.4426950408889634f;
constexpr float CW_NEG_LN2_HI = -0.6931152343750000f;
constexpr float CW_NEG_LN2_LO = -3.19461832987e-05f;

sfpi_inline sfpi::vFloat expm1_cw_clamped(sfpi::vFloat x)
{
    // Clamp to prevent exponent underflow (k < -127 wraps setexp). sfpi::max is an SFPSWAP
    // total-order max, which ranks a sign-set NaN below -87.0 and would replace it with a
    // finite value; keep those lanes as they came in so the result stays NaN. -inf still
    // clamps (expm1(-inf) = -1). The guard is a signed compare of the raw bits against -inf
    // (0xFF800000): of the negative floats only a sign-set NaN exceeds it, so -0, negative
    // finite and -inf stay clamped. Positive lanes need no guard (max already returned them
    // unchanged), so it does not matter that the compare's wrapping subtract misses +inf/+NaN.
    // Cheaper than is_nan, which tests exponent and mantissa separately.
    constexpr std::int32_t negative_infinity_bits = static_cast<std::int32_t>(0xFF800000);
    const sfpi::vFloat x_in                       = x;
    x                                             = sfpi::max(x, -87.0f);
    v_if (sfpi::as<sfpi::vInt>(x_in) > negative_infinity_bits)
    {
        x = x_in;
    }
    v_endif;

    // Cody-Waite range reduction: x = k*ln(2) + r
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    sfpi::vFloat tmp        = x * CW_INV_LN2 + c231;
    sfpi::vFloat k_f        = tmp - c231;
    sfpi::vFloat r          = k_f * CW_NEG_LN2_HI + x;
    r                       = r + k_f * CW_NEG_LN2_LO;

    // expm1(r) = r * h(r), Horner evaluation of h
#ifdef INP_FLOAT32
    sfpi::vFloat h = PolynomialEvaluator::eval(r, 1.0f, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, 8.3691505715e-03f, 1.3948583510e-03f);
#else
    sfpi::vFloat h = PolynomialEvaluator::eval(r, 1.0f, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f);
#endif
    h = r * h;

    // Reconstruct: exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
    // 0x4B3FFF81 = 0x4B400000 - 127: fuses k_int ISUB + bias IADD into a single ISUB
    constexpr int kC231Bias = 0x4B3FFF81;
    sfpi::vFloat two_k      = sfpi::setexp(1.0f, sfpi::as<sfpi::vInt>(tmp) - kC231Bias);
    return (two_k - 1.0f) + two_k * h;
}

} // namespace ckernel::sfpu
