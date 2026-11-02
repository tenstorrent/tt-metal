// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gelu_alt_approx bench: alternative SFPU implementations of tanh-GELU (torch gelu(approximate="tanh"))
// for an fp32 DEST (fp32_dest_acc_en=True). Raw sfpi on purpose: this bypasses the
// gelu_tanh_tile_pack / calculate_gelu_tanh_fast helper to test different instruction sequences for the
// same function at the same precision contract.
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"  // _float_to_int32_for_exp_21f_
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpi.h"

#include "gelu_coeffs.h"

namespace gelu_alt {

using namespace sfpi;
using ckernel::sfpu::PolynomialEvaluator;

// 1/den for den >= 1 from approx_recip (rel err <= 5.6e-3, probe.json). One step: r0 + r0 * (K - den * r0)
// leaves -eps0^2; K = 1 + 2^-16 cancels it exactly at den = 1 (approx_recip(1) = 1 - 2^-8), which is every
// x >~ 4, so gelu(x) = x there. Two steps: the plain r * (2 - den * r) twice, as in base.
template <int NEWTON>
sfpi_inline vFloat recip_ge1(vFloat den, vFloat k_newton) {
    vFloat r = approx_recip(den);
    if constexpr (NEWTON == 1) {
        r = r * (k_newton - den * r) + r;
    } else {
        r = r * (k_newton - den * r);
        r = r * (k_newton - den * r);
    }
    return r;
}

// -------------------------------------------------------------------------------------------------
// base: verbatim copy of calculate_gelu_tanh_fast<true> (blackhole ckernel_sfpu_gelu.h).
// -------------------------------------------------------------------------------------------------
constexpr float BASE_EXP2_C1 = 8.2622300113e-08f;
constexpr float BASE_EXP2_C2 = 3.4278135885e-15f;
constexpr float BASE_EXP2_C3 = 8.8508325543e-23f;
constexpr float BASE_EXP2_C4 = 2.7117093617e-30f;
constexpr float BASE_A = -2.3022081852e+00f;
constexpr float BASE_B = -1.0294324160e-01f;

template <int ITERATIONS = 8>
inline void gelu_base() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat x2 = x * x;
        vFloat t = x2 * vConstFloatPrgm1 + vConstFloatPrgm0;
        vFloat xlog2 = x * t + 127.0f;
        xlog2 = clamp(xlog2, 0.0f, 254.0f);
        vFloat z = as<vFloat>(ckernel::sfpu::_float_to_int32_for_exp_21f_(xlog2));
        vInt exponential_part = exexp(z, ExponentMode::Biased);
        vMag fractional_part = exman(z);
        vFloat frac = convert<vFloat>(fractional_part, RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(frac, 1.0f, vConstFloatPrgm2, BASE_EXP2_C2, BASE_EXP2_C3, BASE_EXP2_C4);
        vFloat e = setexp(frac, exponential_part);
        vFloat den = e + 1.0f;
        vFloat r = approx_recip(den);
        r = r * (2.0f - den * r);
        r = r * (2.0f - den * r);
        dst_reg[0] = x * r;
        dst_reg++;
    }
}

inline void gelu_base_init() {
    vConstFloatPrgm0 = BASE_A;
    vConstFloatPrgm1 = BASE_B;
    vConstFloatPrgm2 = BASE_EXP2_C1;
}

// -------------------------------------------------------------------------------------------------
// t21: same exp_21f split as base, but a degree-3 2^f (zero-mean, BF16 c2/c3), ONE Newton step with a
// bias-cancelling constant (r1 = r0 + r0 * (K - den * r0), K = 1 + E[eps0^2]), constants hoisted.
// -------------------------------------------------------------------------------------------------
template <int NEWTON, int ITERATIONS = 8>
inline void gelu_t21() {
    const vFloat c127 = 127.0f;
    const vFloat c254 = 254.0f;
    const vFloat k_newton = NEWTON == 1 ? NEWTON1_K : 2.0f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat x2 = x * x;
        vFloat t = x2 * vConstFloatPrgm1 + vConstFloatPrgm0;
        vFloat xlog2 = x * t + c127;
        xlog2 = min(max(xlog2, 0.0f), c254);
        vFloat z = as<vFloat>(ckernel::sfpu::_float_to_int32_for_exp_21f_(xlog2));
        vInt exponential_part = exexp(z, ExponentMode::Biased);
        vMag fractional_part = exman(z);
        vFloat frac = convert<vFloat>(fractional_part, RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(frac, 1.0f, vConstFloatPrgm2, T21_C2, T21_C3);
        vFloat e = setexp(frac, exponential_part);
        vFloat den = e + 1.0f;
        dst_reg[0] = x * recip_ge1<NEWTON>(den, k_newton);
        dst_reg++;
    }
}

inline void gelu_t21_init() {
    vConstFloatPrgm0 = BASE_A;
    vConstFloatPrgm1 = BASE_B;
    vConstFloatPrgm2 = T21_C1;
}

// -------------------------------------------------------------------------------------------------
// mag: clamp x, then split log2(e) = n + f with the 1.5*2^23 magic-number add (RNE leaves n = round,
// f in [-0.5, 0.5]); 2^f degree-3; 2^n folded in by an integer add of n << 23 to the exponent field.
// Same single bias-cancelled Newton reciprocal as t21.
// -------------------------------------------------------------------------------------------------
template <int NEWTON, int ITERATIONS = 8>
inline void gelu_mag() {
    const vFloat xc_hi = MAG_XC;
    const vFloat magic = MAG_M;
    const vFloat k_newton = NEWTON == 1 ? NEWTON1_K : 2.0f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat xc = min(max(x, -xc_hi), xc_hi);
        vFloat x2 = xc * xc;
        vFloat t = x2 * vConstFloatPrgm1 + vConstFloatPrgm0;
        vFloat tt = xc * t + magic;
        vFloat neg_n = magic - tt;
        vFloat f = xc * t + neg_n;
        vFloat q = PolynomialEvaluator::eval(f, MAG_C0, vConstFloatPrgm2, MAG_C2, MAG_C3);
        vInt scale = shft(as<vInt>(tt), 23, ShiftMode::Logical);
        vFloat e = as<vFloat>(as<vInt>(q) + scale);
        vFloat den = e + 1.0f;
        dst_reg[0] = x * recip_ge1<NEWTON>(den, k_newton);
        dst_reg++;
    }
}

inline void gelu_mag_init() {
    vConstFloatPrgm0 = BASE_A;
    vConstFloatPrgm1 = BASE_B;
    vConstFloatPrgm2 = MAG_C1;
}

// -------------------------------------------------------------------------------------------------
// magb: mag with a cheaper clamp (setsgn(min(|x|, XC), x): no negated bound to keep live), u = xc * t split
// into n / f with two immediate adds (1.5 * 2^23 is BF16-exact), BF16 c2 / c3 (one SFPLOADI each).
// -------------------------------------------------------------------------------------------------
template <int ITERATIONS = 8>
inline void gelu_magb() {
    const vFloat xc_hi = MAG_XC;
    const vFloat k_newton = NEWTON1_K;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat xc = copysgn(min(abs(x), xc_hi), x);
        vFloat x2 = xc * xc;
        vFloat t = x2 * vConstFloatPrgm1 + vConstFloatPrgm0;
        vFloat u = xc * t;
        vFloat tt = u + 12582912.0f;
        vFloat n = tt - 12582912.0f;
        vFloat f = u - n;
        vFloat q = PolynomialEvaluator::eval(f, MAGB_C0, vConstFloatPrgm2, MAGB_C2, MAGB_C3);
        vInt scale = shft(as<vInt>(tt), 23, ShiftMode::Logical);
        vFloat e = as<vFloat>(as<vInt>(q) + scale);
        vFloat den = e + 1.0f;
        dst_reg[0] = x * recip_ge1<1>(den, k_newton);
        dst_reg++;
    }
}

inline void gelu_magb_init() {
    vConstFloatPrgm0 = BASE_A;
    vConstFloatPrgm1 = BASE_B;
    vConstFloatPrgm2 = MAGB_C1;
}

// -------------------------------------------------------------------------------------------------
// t21s: t21 with the exp_21f split (exexp, exman, shft, exexp, exman, cast) replaced by a truncating
// SFPSTOCHRND fp32 -> uint16 (n = floor(xlog2), the biased exponent) + SFPCAST back + one subtract
// (f = xlog2 - n in [0, 1)). Both clamps are needed: a large negative xlog2 does not convert to 0.
// -------------------------------------------------------------------------------------------------
template <bool HOIST127 = false, int ITERATIONS = 8>
inline void gelu_t21s() {
    const vFloat c254 = 254.0f;
    const vFloat k_newton = NEWTON1_K;
    const vFloat c127 = 127.0f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat x2 = x * x;
        vFloat t = x2 * vConstFloatPrgm1 + vConstFloatPrgm0;
        vFloat xlog2 = HOIST127 ? x * t + c127 : x * t + 127.0f;
        xlog2 = min(max(xlog2, 0.0f), c254);
        vUInt16 n = convert<vUInt16>(xlog2, RoundMode::Zero);
        vFloat nf = convert<vFloat>(as<vSMag>(n), RoundMode::Nearest);
        vFloat f = xlog2 - nf;
        vFloat q = PolynomialEvaluator::eval(f, 1.0f, vConstFloatPrgm2, T21S_C2, T21S_C3);
        vFloat e = setexp(q, as<vInt>(n));
        vFloat den = e + 1.0f;
        dst_reg[0] = x * recip_ge1<1>(den, k_newton);
        dst_reg++;
    }
}

inline void gelu_t21s_init() {
    vConstFloatPrgm0 = BASE_A;
    vConstFloatPrgm1 = BASE_B;
    vConstFloatPrgm2 = T21S_C1;
}

// -------------------------------------------------------------------------------------------------
// hwx: Blackhole SFPARECIP exp mode (sfpi::approx_exp) for exp(-2u) + the same Newton reciprocal.
// -------------------------------------------------------------------------------------------------
template <int ITERATIONS = 8>
inline void gelu_hwx() {
    const vFloat k_newton = NEWTON1_K;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat x2 = x * x;
        vFloat t = x2 * vConstFloatPrgm1 + vConstFloatPrgm0;
        vFloat e = approx_exp(x * t);
        vFloat den = e + 1.0f;
        dst_reg[0] = x * recip_ge1<1>(den, k_newton);
        dst_reg++;
    }
}

inline void gelu_hwx_init() {
    vConstFloatPrgm0 = HWX_A;
    vConstFloatPrgm1 = HWX_B;
}

// -------------------------------------------------------------------------------------------------
// probes (fp32 output): raw hardware approx_exp / approx_recip, SFPADD rounding of the magic add.
// -------------------------------------------------------------------------------------------------
template <int ITERATIONS = 8>
inline void probe_exp() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = approx_exp(dst_reg[0]);
        dst_reg++;
    }
}

template <int ITERATIONS = 8>
inline void probe_recip() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = approx_recip(dst_reg[0]);
        dst_reg++;
    }
}

template <int ITERATIONS = 8>
inline void probe_round() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat tt = x + vConstFloatPrgm0;
        dst_reg[0] = tt - vConstFloatPrgm0;
        dst_reg++;
    }
}

inline void probe_round_init() { vConstFloatPrgm0 = 12582912.0f; }
inline void probe_none_init() {}

}  // namespace gelu_alt
