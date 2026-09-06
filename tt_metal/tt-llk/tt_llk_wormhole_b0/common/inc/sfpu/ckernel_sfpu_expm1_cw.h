// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

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
    // Clamp to prevent exponent field wraparound in setexp below: SFPSETEXP only writes
    // the low 8 bits of the exponent, so an unclamped k (k = round(x/ln2)) outside
    // [-127, 127] silently wraps instead of saturating to 0/inf. -87 keeps the lower
    // bound within that range with margin (exp(-87) already saturates to 0 in float32).
    // The upper side is handled by the biased-exponent branch below instead of a
    // symmetric clamp: SFPSETEXP writes a biased exponent (k+127), which wraps once it
    // exceeds the 8-bit field's max of 255 (k > 128, i.e. x > 128*ln2 ~= 88.7228). A plain
    // clamp at that threshold still returns NaN for the sub-range where the biased
    // exponent is exactly 255 but the true result is finite (2^k is +inf there, and
    // "inf + inf*h" is NaN whenever h < 0) -- see the e == 255 branch below.
    x = sfpi::max(x, -87.0f);

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
    sfpi::vInt e            = sfpi::as<sfpi::vInt>(tmp) - kC231Bias; // biased exponent of 2^k

    // e >= 256 (k > 128, x > 128*ln2): true overflow, saturate to +inf like the reference.
    sfpi::vFloat result = std::numeric_limits<float>::infinity();
    v_if (e <= 254)
    {
        // Unchanged path: this grouping avoids catastrophic cancellation near x = 0.
        sfpi::vFloat two_k = sfpi::setexp(1.0f, e);
        result             = (two_k - 1.0f) + two_k * h;
    }
    v_elseif (e == 255)
    {
        // 2^k itself is +inf here (biased exponent 255), but 2^k*(1+h) may still be
        // finite -- computing "inf + inf*h" directly is NaN whenever h < 0. Rescale by
        // one binade (2^(k-1)) so the intermediate stays representable; the dropped "-1"
        // term is below the ULP of 2^127 and is exact here. Genuine overflows (h >= 0
        // in this band, or e >= 256 above) still saturate to +inf.
        sfpi::vFloat two_k_half = sfpi::setexp(1.0f, 254);
        result                  = two_k_half * (2.0f + 2.0f * h);
    }
    v_endif;
    return result;
}

} // namespace ckernel::sfpu
