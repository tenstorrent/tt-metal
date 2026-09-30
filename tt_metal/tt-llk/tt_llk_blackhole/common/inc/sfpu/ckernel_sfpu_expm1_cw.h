// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
// Largest argument whose exp(x)-1 is still a finite fp32: exp(88.5) = 2.82e38,
// exp(88.7229) = FLT_MAX. Above this expm1(x) is +inf.
constexpr float CW_EXPM1_MAX  = 88.5f;

sfpi_inline sfpi::vFloat expm1_cw_clamped(sfpi::vFloat x)
{
    // Clamp to prevent exponent underflow (k < -127 wraps setexp)
    x = sfpi::max(x, -87.0f);

    // ... and on the HIGH side, which was missing. The Cody-Waite reduction has no
    // upper bound: x*CW_INV_LN2 itself overflows to +inf near FLT_MAX (so r = -inf
    // and h = -inf), and setexp writes an 8-bit exponent FIELD that WRAPS instead of
    // saturating, so two_k collapses to 1.0. Both regimes returned a NEGATIVE value
    // where expm1 is +inf: -inf at x = 3.3e38, -3.2e21 at x = 1e10. Clamp the
    // ARGUMENT so the reduction stays in range, then state the tail analytically.
    const sfpi::vFloat x_in = x;
    x                       = sfpi::min(x, CW_EXPM1_MAX);

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
    sfpi::vFloat result     = (two_k - 1.0f) + two_k * h;
    v_if (x_in > CW_EXPM1_MAX)
    {
        result = Converter::as_float(0x7F800000U); // +inf
    }
    v_endif;
    return result;
}

} // namespace ckernel::sfpu
