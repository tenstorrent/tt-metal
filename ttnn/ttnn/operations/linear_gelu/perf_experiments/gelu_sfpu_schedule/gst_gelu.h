// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gelu_sfpu_schedule bake-off: instruction-level variants of calculate_gelu_tanh_fast (x / (1 + exp(-2u)),
// u = sqrt(2/pi)(x + 0.044715 x^3)), selected by GST_VARIANT. Include this header FIRST in a compute kernel:
// for GST_VARIANT > 0 it renames calculate_gelu_tanh_fast / gelu_tanh_fast_init so gelu.h (and through it
// GeluTanhActivation<1> / gelu_tanh_tile_pack) dispatch to the variant here. GST_VARIANT 0 = the stock function.
//
// Raw-sfpi (no helper exists for this level): bypasses ckernel::sfpu::calculate_gelu_tanh_fast to change its
// instruction schedule; the SFPU there is issue/latency bound (32 instr + ~10 RAW stalls per row).
//   1 base_nr  stock exp, Newton steps as r += r(1 - d r) on the 1.0 const reg (drops 2 SFPLOADI of 2.0)
//   2 magic4   exp2 via magic-number round (z/64 + 1.5*2^17): k and r = z - k in 3 ops, 2^k by one shift,
//              clamp on the -1 / 1 const regs (no SFPLOADI), deg-4 2^r (rel 2.8e-6), den = p*2^k + 1 fused,
//              one cubic-convergent reciprocal correction r0 (1 + e + e^2)
//   3 magic3   magic4 with a deg-3 2^r (rel 1.8e-4, zero-mean refit)
//   4 magic4x2 magic4 (bf16 c2..c4), two rows per trip, chains interleaved A/B to fill MAD latency; x reloaded
//   6 magic3x2 same with the deg-3 bf16 polynomial
//   7 / 8      magic4x2 / magic3x2 with a single linear reciprocal correction r0 (1 + e) (one-sided error e^2)
//   10 / 11    magic4x2w / magic3x2w: wide exponent range (scale 128, E up to 2^126) for +1 SFPLOADI per row pair
//   90 / 91    diagnostics: empty body; magic3x2_nr1 without the clamp (wrong for |x| > ~8)
//   5 hwexp    SFPARECIP exp mode (approx_exp) for exp(-2u) directly; precision probe

#pragma once

#include "api/compute/common_globals.h"

#ifndef GST_VARIANT
#define GST_VARIANT 0
#endif

#if (defined(TRISC_MATH) || defined(TRISC_PACK)) && GST_VARIANT > 0
#include "ckernel_sfpu_gelu.h"

namespace ckernel::sfpu {

// log2 of exp(-2u) / 64 = x (A + B x^2) / 64
constexpr float GST_A64 = -2.3022081852e+00f / 64.0f;
constexpr float GST_B64 = -1.0294324160e-01f / 64.0f;
// 1.5 * 2^17 (ulp 2^-6, so the add rounds w to a multiple of 1/64) + 127/64 (biased exponent in the low bits).
constexpr float GST_MAGIC = 196608.0f + 127.0f / 64.0f;

// 2^r, r = 64 r', as 1 + c1 r' + ... (coefficients scaled by 64^k). Fitted in fit_exp2.py.
constexpr float GST_D4_C1 = 4.4359943390e+01f;
constexpr float GST_D4_C2 = 9.8402740479e+02f;
constexpr float GST_D4_C3 = 1.4655639648e+04f;
constexpr float GST_D4_C4 = 1.6076895312e+05f;
constexpr float GST_D3_C1 = 4.4414344788e+01f;
constexpr float GST_D3_C2 = 9.9067388916e+02f;
constexpr float GST_D3_C3 = 1.3634805664e+04f;

// bf16 c2.. (one SFPLOADI each), fp32 c1 in vConstFloatPrgm2, zero-mean minimax (fit_exp2.py bf16 refit).
constexpr float GST_B4_C1 = 4.4360542297e+01f;
constexpr float GST_B4_C2 = 9.8400000000e+02f;
constexpr float GST_B4_C3 = 1.4656000000e+04f;
constexpr float GST_B4_C4 = 1.6179200000e+05f;
constexpr float GST_B3_C1 = 4.4445056915e+01f;
constexpr float GST_B3_C2 = 9.9200000000e+02f;
constexpr float GST_B3_C3 = 1.3632000000e+04f;

// WIDE (scale 128): x (A + B x^2) / 128; 1.5 * 2^16 (ulp 2^-7) + 128/128; scale-128 bf16 c2.., zero-mean.
constexpr float GST_A128 = -2.3022081852e+00f / 128.0f;
constexpr float GST_B128 = -1.0294324160e-01f / 128.0f;
// Bias 146 with a -19/128 offset on w: field = round(128 w) + 146 in [18, 253], E = 2^z on [2^-109, 2^126];
// den >= 2^126 makes approx_recip return 0, so the deep negative tail is exactly 0 like the stock function.
constexpr float GST_W_MAGIC = 98304.0f + 146.0f / 128.0f;
constexpr float GST_W_OFF = -19.0f / 128.0f;
constexpr float GST_W4_C1 = 8.8721084595e+01f;
constexpr float GST_W4_C2 = 3.9360000000e+03f;
constexpr float GST_W4_C3 = 1.1724800000e+05f;
constexpr float GST_W4_C4 = 2.5886720000e+06f;
constexpr float GST_W3_C1 = 8.8890113831e+01f;
constexpr float GST_W3_C2 = 3.9680000000e+03f;
constexpr float GST_W3_C3 = 1.0905600000e+05f;

// exp(-2u) for hwexp: y = x (A + B x^2) in natural log units
constexpr float GST_YA = -1.5957691216e+00f;  // -2 sqrt(2/pi)
constexpr float GST_YB = -1.5957691216e+00f * 0.044715f;

// One row: returns GELU(x) via the magic-number exp2. DEG selects the 2^r polynomial.
template <int DEG>
sfpi_inline sfpi::vFloat gst_magic_row(
    sfpi::vFloat x, sfpi::vFloat magic, sfpi::vFloat c2, sfpi::vFloat c3, sfpi::vFloat c4) {
    sfpi::vFloat t = x * x;
    t = t * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0;
    sfpi::vFloat w = x * t;
    // |z| <= 64 keeps the biased exponent in [63, 191]; the bounds are the -1 / 1 constant registers.
    w = sfpi::max(w, sfpi::vFloat(-1.0f));
    w = sfpi::min(w, sfpi::vFloat(1.0f));
    sfpi::vFloat kb = w + magic;
    sfpi::vFloat kf = kb - magic;  // round(64 w) / 64, exact
    sfpi::vFloat r = w - kf;       // in [-1/128, 1/128]
    sfpi::vFloat s =
        sfpi::as<sfpi::vFloat>(sfpi::shft(sfpi::as<sfpi::vUInt>(kb), 23));  // low 9 bits (64 w + 127) -> exponent field
    sfpi::vFloat p;
    if constexpr (DEG == 4) {
        p = c4 * r + c3;
        p = p * r + c2;
    } else {
        p = c3 * r + c2;
    }
    p = p * r + sfpi::vConstFloatPrgm2;
    p = p * r + 1.0f;
    sfpi::vFloat den = p * s + 1.0f;
    sfpi::vFloat r0 = sfpi::approx_recip(den);
    // 1/den = r0 / (1 - e) = r0 (1 + e + e^2 + ...), e = 1 - den r0; truncation error e^3.
    sfpi::vFloat e = 1.0f - den * r0;
    sfpi::vFloat q = e * e + e;
    sfpi::vFloat xr = x * r0;
    return xr * q + xr;
}

// Two rows per trip, statements interleaved A/B so no instruction reads the previous one's result (BH SFPU
// stalls one cycle on a back-to-back dependency). 3 LRegs per chain (x is reloaded from DEST for the final
// multiply); the other 2 hold M and one rotating bf16 coefficient slot shared by both chains.
// WIDE: w = z/128 - 19/128 clamped to the [-1, 0.8373] constant registers, field = round(128 w) + 146, so
// E = 2^z covers [2^-109, 2^126] (narrow: z/64 on [-1, 1], E <= 2^64, wrong |GELU| for x < -1.8e16).
template <bool D4, bool NR1, bool CLAMP, bool WIDE, int ITERATIONS>
sfpi_inline void gst_x2() {
    sfpi::vFloat magic = WIDE ? GST_W_MAGIC : GST_MAGIC;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat ta = sfpi::dst_reg[0];
        sfpi::vFloat tb = sfpi::dst_reg[1];
        sfpi::vFloat fa = ta * ta;
        sfpi::vFloat fb = tb * tb;
        fa = fa * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0;
        fb = fb * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0;
        if constexpr (WIDE) {
            sfpi::vFloat off = GST_W_OFF;
            ta = ta * fa + off;
            tb = tb * fb + off;
        } else {
            ta = ta * fa;
            tb = tb * fb;
        }
        if constexpr (CLAMP) {
            ta = sfpi::max(ta, sfpi::vFloat(-1.0f));
            tb = sfpi::max(tb, sfpi::vFloat(-1.0f));
            ta = sfpi::min(ta, sfpi::vFloat(WIDE ? 0.837300003f : 1.0f));
            tb = sfpi::min(tb, sfpi::vFloat(WIDE ? 0.837300003f : 1.0f));
        }
        sfpi::vFloat ka = ta + magic;
        sfpi::vFloat kb = tb + magic;
        fa = ka - magic;  // round(s w) / s, exact
        fb = kb - magic;
        ta = ta - fa;  // r' = r / s, r in [-0.5, 0.5]
        tb = tb - fb;
        ka = sfpi::as<sfpi::vFloat>(sfpi::shft(sfpi::as<sfpi::vUInt>(ka), 23));  // low 9 bits -> exponent
        kb = sfpi::as<sfpi::vFloat>(sfpi::shft(sfpi::as<sfpi::vUInt>(kb), 23));
        if constexpr (D4) {
            sfpi::vFloat c4 = WIDE ? GST_W4_C4 : GST_B4_C4;
            sfpi::vFloat c3 = WIDE ? GST_W4_C3 : GST_B4_C3;
            fa = c4 * ta + c3;
            fb = c4 * tb + c3;
            sfpi::vFloat c2 = WIDE ? GST_W4_C2 : GST_B4_C2;
            fa = fa * ta + c2;
            fb = fb * tb + c2;
        } else {
            sfpi::vFloat c3 = WIDE ? GST_W3_C3 : GST_B3_C3;
            sfpi::vFloat c2 = WIDE ? GST_W3_C2 : GST_B3_C2;
            fa = c3 * ta + c2;
            fb = c3 * tb + c2;
        }
        fa = fa * ta + sfpi::vConstFloatPrgm2;
        fb = fb * tb + sfpi::vConstFloatPrgm2;
        fa = fa * ta + 1.0f;
        fb = fb * tb + 1.0f;
        fa = fa * ka + 1.0f;  // den = 1 + 2^r 2^k
        fb = fb * kb + 1.0f;
        ka = sfpi::approx_recip(fa);
        kb = sfpi::approx_recip(fb);
        // 1/den = r0 / (1 - e) = r0 (1 + e + e^2 + ...), e = 1 - den r0: cubic error e^3 (NR1: e^2, one-sided)
        fa = 1.0f - fa * ka;
        fb = 1.0f - fb * kb;
        if constexpr (!NR1) {
            fa = fa * fa + fa;
            fb = fb * fb + fb;
        }
        ta = sfpi::dst_reg[0];
        tb = sfpi::dst_reg[1];
        ta = ta * ka;
        tb = tb * kb;
        ta = ta * fa + ta;
        tb = tb * fb + tb;
        sfpi::dst_reg[0] = ta;
        sfpi::dst_reg[1] = tb;
        sfpi::dst_reg += 2;
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void gst_calculate_gelu_tanh_fast() {
#if GST_VARIANT == 1
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0;
        sfpi::vFloat xlog2 = x * t + 127.0f;
        xlog2 = sfpi::clamp(xlog2, 0.0f, 254.0f);
        sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
        sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part = sfpi::exman(z);
        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(
            frac, 1.0f, sfpi::vConstFloatPrgm2, GELU_TANH_FAST_EXP2_C2, GELU_TANH_FAST_EXP2_C3, GELU_TANH_FAST_EXP2_C4);
        sfpi::vFloat e = sfpi::setexp(frac, exponential_part);
        sfpi::vFloat den = e + 1.0f;
        sfpi::vFloat r = sfpi::approx_recip(den);
        sfpi::vFloat err = 1.0f - den * r;
        r = r * err + r;
        err = 1.0f - den * r;
        r = r * err + r;
        sfpi::vFloat result = x * r;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
#elif GST_VARIANT == 2 || GST_VARIANT == 3
    constexpr int DEG = GST_VARIANT == 2 ? 4 : 3;
    sfpi::vFloat magic = GST_MAGIC;
    sfpi::vFloat c2 = DEG == 4 ? GST_D4_C2 : GST_D3_C2;
    sfpi::vFloat c3 = DEG == 4 ? GST_D4_C3 : GST_D3_C3;
    sfpi::vFloat c4 = GST_D4_C4;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = gst_magic_row<DEG>(x, magic, c2, c3, c4);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
#elif GST_VARIANT == 4 || GST_VARIANT == 6 || GST_VARIANT == 7 || GST_VARIANT == 8 || GST_VARIANT == 91 || \
    GST_VARIANT == 10 || GST_VARIANT == 11
    // linear_gelu only supports fp32 DEST; the bf16-DEST arm (SFPSTOCHRND to bf16 before the store) is not here.
    static_assert(is_fp32_dest_acc_en, "gst_x2 variants assume fp32 DEST");
    static_assert(ITERATIONS % 2 == 0, "two rows per trip");
    constexpr int V = GST_VARIANT;
    gst_x2<V == 4 || V == 7 || V == 10, V == 7 || V == 8 || V == 91, V != 91, V == 10 || V == 11, ITERATIONS>();
#elif GST_VARIANT == 90
    // diagnostic: no SFPU math, walk DEST only (per-call / per-face overhead floor)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg++;
    }
#elif GST_VARIANT == 5
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat t = x * x;
        t = t * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0;
        sfpi::vFloat y = x * t;
        sfpi::vFloat den = sfpi::approx_exp(y) + 1.0f;
        sfpi::vFloat r0 = sfpi::approx_recip(den);
        sfpi::vFloat e = 1.0f - den * r0;
        sfpi::vFloat q = e * e + e;
        sfpi::vFloat xr = x * r0;
        sfpi::dst_reg[0] = xr * q + xr;
        sfpi::dst_reg++;
    }
#endif
}

inline void gst_gelu_tanh_fast_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#if GST_VARIANT == 1
    sfpi::vConstFloatPrgm0 = GELU_TANH_FAST_A;
    sfpi::vConstFloatPrgm1 = GELU_TANH_FAST_B;
    sfpi::vConstFloatPrgm2 = GELU_TANH_FAST_EXP2_C1;
#elif GST_VARIANT == 5
    sfpi::vConstFloatPrgm0 = GST_YA;
    sfpi::vConstFloatPrgm1 = GST_YB;
#elif GST_VARIANT == 10 || GST_VARIANT == 11
    sfpi::vConstFloatPrgm0 = GST_A128;
    sfpi::vConstFloatPrgm1 = GST_B128;
    sfpi::vConstFloatPrgm2 = GST_VARIANT == 10 ? GST_W4_C1 : GST_W3_C1;
#else
    sfpi::vConstFloatPrgm0 = GST_A64;
    sfpi::vConstFloatPrgm1 = GST_B64;
    sfpi::vConstFloatPrgm2 = GST_VARIANT == 2                         ? GST_D4_C1
                             : GST_VARIANT == 3                       ? GST_D3_C1
                             : (GST_VARIANT == 4 || GST_VARIANT == 7) ? GST_B4_C1
                                                                      : GST_B3_C1;
#endif
}

}  // namespace ckernel::sfpu

#define calculate_gelu_tanh_fast gst_calculate_gelu_tanh_fast
#define gelu_tanh_fast_init gst_gelu_tanh_fast_init
#endif
