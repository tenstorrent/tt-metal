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

// The two highest-order Horner coefficients of h(r), exported so a row-loop caller can hold them in LRegs
// (see the hoisted-constant overload below). Which polynomial they belong to follows the build, as the
// evaluation itself does.
#ifdef INP_FLOAT32
constexpr float CW_EXPM1_H_TOP1 = 8.3691505715e-03f;
constexpr float CW_EXPM1_H_TOP0 = 1.3948583510e-03f;
#else
constexpr float CW_EXPM1_H_TOP1 = 4.1875664145e-02f;
constexpr float CW_EXPM1_H_TOP0 = 8.3751315251e-03f;
#endif

// expm1_cw_clamped with its five loop-invariant fp32 constants supplied by the caller. Each may be a float
// (materialised by SFPLOADI where used, exactly as the literal is in expm1_cw_clamped(x)) or an
// sfpi::vFloat / vConstFloatPrgmN the caller holds across its row loop: sfpi 7.83.0 never hoists a literal
// out of a loop by itself, so in a row loop every fp32 literal otherwise costs an SFPLOADI pair per row.
// Identical arithmetic either way. The constants are template-typed rather than sfpi::vFloat because a
// vFloat parameter is built before the call and stays live through the body, which changes the
// non-hoisting callers' code.
template <typename K, typename HI, typename LO, typename T1, typename T0>
sfpi_inline sfpi::vFloat expm1_cw_clamped(sfpi::vFloat x, K inv_ln2, HI neg_ln2_hi, LO neg_ln2_lo, T1 h_top1, T0 h_top0)
{
    // Clamp to prevent exponent underflow (k < -127 wraps setexp)
    x = sfpi::max(x, -87.0f);

    // Cody-Waite range reduction: x = k*ln(2) + r
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    sfpi::vFloat tmp        = x * inv_ln2 + c231;
    sfpi::vFloat k_f        = tmp - c231;
    sfpi::vFloat r          = k_f * neg_ln2_hi + x;
    r                       = r + k_f * neg_ln2_lo;

    // expm1(r) = r * h(r), Horner evaluation of h
#ifdef INP_FLOAT32
    sfpi::vFloat h = PolynomialEvaluator::eval(r, 1.0f, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, h_top1, h_top0);
#else
    sfpi::vFloat h = PolynomialEvaluator::eval(r, 1.0f, 4.9999371171e-01f, 1.6666433215e-01f, h_top1, h_top0);
#endif
    h = r * h;

    // Reconstruct: exp(x)-1 = (2^k - 1) + 2^k * expm1(r)
    // 0x4B3FFF81 = 0x4B400000 - 127: fuses k_int ISUB + bias IADD into a single ISUB
    constexpr int kC231Bias = 0x4B3FFF81;
    sfpi::vFloat two_k      = sfpi::setexp(1.0f, sfpi::as<sfpi::vInt>(tmp) - kC231Bias);
    return (two_k - 1.0f) + two_k * h;
}

sfpi_inline sfpi::vFloat expm1_cw_clamped(sfpi::vFloat x)
{
    return expm1_cw_clamped(x, CW_INV_LN2, CW_NEG_LN2_HI, CW_NEG_LN2_LO, CW_EXPM1_H_TOP1, CW_EXPM1_H_TOP0);
}

// Row-loop form for kernels whose init programmed the Cody-Waite constants into vConstFloatPrgm0/1/2 with
// expm1_cw_init_prgm_consts(). The two top Horner coefficients come from the caller: an sfpi::vFloat built
// once before its loop, or a float literal where its LRegs are all spoken for (sfpi cannot spill: one
// hoisted constant too many is a compile error, "too few lregs", never a slowdown). Only for kernels that
// run no reciprocal: Prgm0 is sfpu_reciprocal_init's 2.0f wherever one does.
inline void expm1_cw_init_prgm_consts()
{
    sfpi::vConstFloatPrgm0 = CW_INV_LN2;
    sfpi::vConstFloatPrgm1 = CW_NEG_LN2_HI;
    sfpi::vConstFloatPrgm2 = CW_NEG_LN2_LO;
}

template <typename T1, typename T0>
sfpi_inline sfpi::vFloat expm1_cw_clamped_prgm(sfpi::vFloat x, T1 h_top1, T0 h_top0)
{
    return expm1_cw_clamped(x, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2, h_top1, h_top0);
}

} // namespace ckernel::sfpu
