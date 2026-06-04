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

sfpi_inline sfpi::vFloat expm1_cw_clamped(sfpi::vFloat x)
{
    // Clamp to prevent exponent underflow (k < -127 wraps setexp)
    sfpi::vFloat lo = -87.0f;
    sfpi::vec_min_max(lo, x);

    // Cody-Waite range reduction: x = k*ln(2) + r
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    sfpi::vFloat tmp        = x * CW_INV_LN2 + c231;
    sfpi::vFloat k_f        = tmp - c231;
    sfpi::vFloat r          = k_f * CW_NEG_LN2_HI + x;
    r                       = r + k_f * CW_NEG_LN2_LO;

    // expm1(r) = r * h(r), Horner evaluation of h
#ifdef INP_FLOAT32
    sfpi::vFloat h = PolynomialEvaluator::eval(r, sfpi::vConst1, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, 8.3691505715e-03f, 1.3948583510e-03f);
#else
    sfpi::vFloat h = PolynomialEvaluator::eval(r, sfpi::vConst1, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f);
#endif
    h = r * h;

    // Reconstruct: exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
    // 0x4B3FFF81 = 0x4B400000 - 127: fuses k_int ISUB + bias IADD into a single ISUB
    constexpr int kC231Bias = 0x4B3FFF81;
    sfpi::vFloat two_k      = sfpi::setexp(sfpi::vConst1, sfpi::reinterpret<sfpi::vInt>(tmp) - kC231Bias);
    return (two_k - sfpi::vConst1) + two_k * h;
}

} // namespace ckernel::sfpu
