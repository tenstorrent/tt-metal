// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Generic embedded piecewise-LUT SFPU evaluator for the unary_lut DFB op.
//
// Supports two evaluation methods, selected by LUT_EVAL_METHOD:
//   0 (POLY_CASCADE): per-segment Horner polynomial of degree LUT_POLY_DEGREE.
//   1 (RATIONAL):     per-segment P(x)/Q(x), numerator degree LUT_NUM_DEGREE,
//                     denominator degree LUT_DEN_DEGREE.
// Both handle single-segment (NUM_SEGMENTS==1) and multi-segment cascades. NO range
// reduction in this slice (the deployed configs that need it are out of scope here).
//
// This is the DFB analog of the proven Quasar generic-LUT SFPU eval in
//   tt_metal/tt-llk/tests/sources/quasar/generic_lut_activation_quasar_test.cpp.
// The LUT (boundaries + per-segment coefficients) is baked in at compile time via
// the LUT_* defines emitted by the program factory (UnaryLutDeviceOperation passes
// them as -D defines so the JIT bakes a per-activation kernel; no host rebuild).
//
// LUT_DATA layout:
//   POLY:     [b0 .. bS  (NUM_SEGMENTS+1 ascending boundaries),
//              then per segment (POLY_DEGREE+1) Horner coeffs c0 .. cN]
//   RATIONAL: [b0 .. bS,
//              then per segment (NUM_DEGREE+1) numerator coeffs n0..nN
//                              + (DEN_DEGREE+1) denominator coeffs d0..dM]
//
// Quasar-specific constraints baked in (proven by the tt-llk test):
//   * Horner step uses a fused SFPMAD (`fma_const`) — Quasar removed SFPADDI, so a
//     standalone `acc*x + const` would lower to an aborted SFPADDI.
//   * Each segment's argument is clamped to that segment's own [b_seg, b_{seg+1}]
//     sub-interval BEFORE evaluation (top-level v_if, never nested), so no
//     polynomial is ever evaluated outside its fit range (sfpi runs the RHS on ALL
//     32 lanes regardless of the predicate).
//   * Coefficients are read through COMPILE-TIME constexpr indices so the backend
//     folds them into SFPLOADI immediates and elides the LUT_DATA array.
//   * COMPARE-CONSTANT BUG WORKAROUND (the 0.93->>=0.99 fix): see `sub()` note below.
//
// Called via _llk_math_eltwise_unary_sfpu_params_(calculate_lut_activation<ITER>, dst).

#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "sfpi.h"
#ifdef ARCH_QUASAR
#include "cmath_common.h"  // ckernel::math::_incr_counters_ / SFP_ROWS
#endif

// ---- LUT compile-time configuration (emitted by the program factory) --------
#ifndef LUT_EVAL_METHOD
#define LUT_EVAL_METHOD 0  // 0 = POLY_CASCADE, 1 = RATIONAL
#endif
#ifndef LUT_POLY_DEGREE
#define LUT_POLY_DEGREE 2
#endif
#ifndef LUT_NUM_SEGMENTS
#define LUT_NUM_SEGMENTS 4
#endif
#ifndef LUT_NUM_DEGREE
#define LUT_NUM_DEGREE 0
#endif
#ifndef LUT_DEN_DEGREE
#define LUT_DEN_DEGREE 0
#endif
#ifndef LUT_DATA_INIT
// clang-format off
// Default: the proven deg-2 / 4-seg sigmoid LUT (no range reduction). Matches the
// tt-llk Quasar test default and DEFAULT_COEFFS / DEFAULT_BOUNDARIES in the python
// golden, so PCC isolates the DFB-path correctness.
#define LUT_DATA_INIT {                       \
    /* boundaries b0..b4 */                   \
    -4.0f, -2.0f, 0.0f, 2.0f, 4.0f,           \
    /* seg0 coeffs: c0, c1, c2 */             \
    0.38296354f, 0.17515847f, 0.02109685f,    \
    /* seg1 */                                \
    0.50329190f, 0.27505103f, 0.04113654f,    \
    /* seg2 */                                \
    0.49670810f, 0.27505103f, -0.04113654f,   \
    /* seg3 */                                \
    0.61703646f, 0.17515847f, -0.02109685f }
// clang-format on
#endif

// ---- Range-reduction configuration (emitted by the program factory) ---------
// The factory parses range_reduction_method (+ params) from the fitter CSV METADATA
// into the LutConfig and emits these as -D defines (see make_lut_defines). The kernel
// then does the exact reduce-then-poly-then-reconstruct from range_reduction.py, so
// the golden is the TRUE activation over the FULL original domain. With no RR
// (LUT_RR_METHOD == 0, the default) the reduce/reconstruct blocks compile away and the
// kernel is byte-identical to the no-RR poly/rational path. Method codes + constants
// mirror the tt-llk Quasar generic-LUT reference exactly.
#ifndef LUT_RR_METHOD
#define LUT_RR_METHOD \
    0  // 0 none, 1 log, 2 exp, 3 cbrt, 4 expalu_exp2,
       // 5 expalu_log2, 6 expalu_pow, 7 trig (sin/cos), 8 tan,
       // 9 newton_root (magic-seed + Newton/Householder; STANDALONE evaluator,
       //   bypasses the segment cascade — sqrt / rsqrt / cbrt)
#endif

// ---- Newton-root configuration (method 9). The fitter's deployed sqrt / rsqrt picks
// use eval_method == newton_root: a magic-number bit-hack seed followed by a fixed
// number of Newton/Householder iterations. This is a SELF-CONTAINED evaluator over the
// FULL input domain — the magic seed already folds in the exponent decomposition, so
// there is NO reduce / segment-select / reconstruct (the CSV carries a single trivial
// segment). Constants are emitted by the factory from the fitter CSV METADATA
// (newton_root_magic / _c1 / _c2 / _n / _iters / _reciprocal). Ported verbatim from
// the proven tt-llk reference generic_lut_newton_root_quasar_test.cpp.
#ifndef LUT_NR_MAGIC
#define LUT_NR_MAGIC 0x5f1110a0  // sqrt magic seed (default); rsqrt uses 0x5f3759df
#endif
#ifndef LUT_NR_C1
#define LUT_NR_C1 2.2825186f
#endif
#ifndef LUT_NR_C2
#define LUT_NR_C2 2.2533049f
#endif
#ifndef LUT_NR_ITERS
#define LUT_NR_ITERS 2
#endif
#ifndef LUT_NR_N
#define LUT_NR_N 2  // root order (2 = sqrt/rsqrt)
#endif
#ifndef LUT_NR_RECIPROCAL
#define LUT_NR_RECIPROCAL 0  // 0 = sqrt, 1 = rsqrt
#endif
#ifndef LUT_RR_LOG_LN2
#define LUT_RR_LOG_LN2 1.0f
#endif
#ifndef LUT_RR_EXP_MULT
#define LUT_RR_EXP_MULT 1.4426950408889634f
#endif
#ifndef LUT_RR_EXP_CONST
#define LUT_RR_EXP_CONST 0.6931471805599453f
#endif
#ifndef LUT_RR_SCALE0
#define LUT_RR_SCALE0 1.0f
#endif
#ifndef LUT_RR_SCALE1
#define LUT_RR_SCALE1 1.0f
#endif
#ifndef LUT_RR_SCALE2
#define LUT_RR_SCALE2 1.0f
#endif
#ifndef LUT_RR_EXP2_MULT
#define LUT_RR_EXP2_MULT 1.0f
#endif
#ifndef LUT_RR_COMPOSE
#define LUT_RR_COMPOSE 0  // 0 none, 1 sigmoid, 2 minus_one
#endif
#ifndef LUT_RR_LOG2_SCALE
#define LUT_RR_LOG2_SCALE 1.0f
#endif
#ifndef LUT_RR_LOG2_BASIS_MMINUS1
#define LUT_RR_LOG2_BASIS_MMINUS1 0
#endif
#ifndef LUT_RR_INPUT_OFFSET
#define LUT_RR_INPUT_OFFSET 0.0f
#endif
#ifndef LUT_RR_POW_N
#define LUT_RR_POW_N 2
#endif
#ifndef LUT_RR_POW_RECIP
#define LUT_RR_POW_RECIP 0
#endif
// trig (method 7) / tan (method 8): the kernel hardcodes pi / Cody-Waite constants
// (matching range_reduction.py); they are fp32-exact and carry no per-CSV params.
#ifndef LUT_RR_TRIG_INV_PI
#define LUT_RR_TRIG_INV_PI 0.3183098861837907f  // 1/pi
#endif
#ifndef LUT_RR_TRIG_PI
#define LUT_RR_TRIG_PI 3.141592653589793f
#endif
#ifndef LUT_RR_TAN_INV_HALFPI
#define LUT_RR_TAN_INV_HALFPI 0.6366197723675814f  // 1/(pi/2)
#endif
#ifndef LUT_RR_TAN_PI2_HI
#define LUT_RR_TAN_PI2_HI 1.5703125f  // pi/2 high bits
#endif
#ifndef LUT_RR_TAN_PI2_LO
#define LUT_RR_TAN_PI2_LO 0.0004837512969970703f  // pi/2 remainder
#endif

namespace ckernel {
namespace sfpu {

namespace lut_detail {

constexpr std::uint32_t EVAL_METHOD = LUT_EVAL_METHOD;
constexpr std::uint32_t POLY_DEGREE = LUT_POLY_DEGREE;
constexpr std::uint32_t NUM_SEGMENTS = LUT_NUM_SEGMENTS;
constexpr std::uint32_t NUM_DEGREE = LUT_NUM_DEGREE;
constexpr std::uint32_t DEN_DEGREE = LUT_DEN_DEGREE;

// Per-segment coefficient stride: POLY -> degree+1; RATIONAL -> (num+1)+(den+1).
constexpr std::uint32_t COEFFS_PER_SEGMENT =
    (EVAL_METHOD == 1) ? ((NUM_DEGREE + 1) + (DEN_DEGREE + 1)) : (POLY_DEGREE + 1);
constexpr std::uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // skip boundaries
constexpr std::uint32_t LUT_SIZE = COEFF_OFFSET + NUM_SEGMENTS * COEFFS_PER_SEGMENT;

constexpr std::array<float, LUT_SIZE> LUT_DATA = LUT_DATA_INIT;

// ---------------------------------------------------------------------------
// CRAQ-SIM QUASAR -1.0 CONSTANT-REGISTER BUG WORKAROUND.
//
// sfpi maps a few float constants to dedicated "constant LRegs" (CREGs) instead of an
// SFPLOADI: CREG 9 = 0.0, CREG 10 = +1.0, CREG 11 = -1.0 (CREG_IDX_NEG_1 ==
// CREG_IDX_PRGM0 — a PROGRAMMABLE const reg loaded via the SFPCONFIG path). The metal
// JIT -O2 backend substitutes the matching CREG for any literal `vFloat(-1.0f)` (and
// re-derives it even from an int-reinterpret or split SFPLOADI), saving the load. On
// real silicon the LLK prologue programs CREG 11 = -1.0; craq-sim's reset only inits
// CREG 8 / 10 / 15 (tensix.cpp ~L100-108) and leaves CREG 11 reading 0. So ANY
// coefficient of exactly -1.0 (common in normalized minimax RATIONAL denominators —
// atanh n6d6 has d2 = -1.0; and trivial linear segments like abs have a leading -1.0)
// silently evaluates as 0, dropping the term (proven by probe: p(x)=x-1 yields x).
//
// FIX: program CREG 11 = -1.0 ourselves at SFPU-callback entry, exactly as silicon's
// `_sfpu_load_config32_(11, ...)` does. Then the backend's CREG substitution reads the
// correct -1.0. General (fixes every -1.0 use), principled (it is the real init), and
// uses no coefficient special-casing. const_vf() then just emits `vFloat(c)` as usual.
sfpi_inline void init_neg1_const_reg() {
#ifdef ARCH_QUASAR
    // dest = CREG_IDX_NEG_1 (11); -1.0f == 0xBF800000 -> upper16 0xBF80, lower16 0x0000.
    ckernel::math::_sfpu_load_config32_(11, 0xBF80u, 0x0000u);
#endif
}

sfpi_inline sfpi::vFloat const_vf(float c) { return sfpi::vFloat(c); }

// Fused multiply-add with a compile-time addend (SFPADDI avoidance, see header note).
__attribute__((always_inline)) sfpi_inline sfpi::vFloat fma_const(sfpi::vFloat a, sfpi::vFloat b, float c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), const_vf(c).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// Horner FMA step: acc*x + c. Passes the runtime register `x` as the SFPMAD multiplicand.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat horner_fma(sfpi::vFloat acc, sfpi::vFloat x, float c) {
    return __builtin_rvtt_sfpmad(x.get(), acc.get(), const_vf(c).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// ---------------------------------------------------------------------------
// CRAQ-SIM QUASAR COMPARE-CONSTANT BUG WORKAROUND (the 0.93->>=0.99 fix).
//
// Proven by in-kernel instrumentation on craq-sim: a sfpi `v_if(a < b)` /
// `v_if(a >= b)` where `b` is a NONZERO compile-time float (materialized as a vFloat
// constant) IGNORES the comparison operand and behaves as `v_if(a < 0)` /
// `v_if(a >= 0)`. The vFloat-vs-vFloat compare lowers on Quasar to a fused
// SFPSWAP+SFPSETCC load-macro sequence; under metal's JIT -O flags the SFPLOADI that
// stages the constant operand is reordered/coalesced so the compare's source register
// reads back 0 (the tt-llk STANDALONE build uses different flags and stages it
// correctly, hence PCC~1 there but 0.93 here). The segment select then collapses to a
// sign(x) step function (seg0 const for x<0, top-seg const for x>=0), the exact
// observed two-valued output.
//
// FIX: never compare against a nonzero constant. Form the difference (a - b) with a
// fused SFPMAD and compare it against literal 0.0f. The intended threshold is now
// encoded in the operand VALUE, not in a compare immediate, so the result is correct
// whether or not the compare's constant operand survives. This is mathematically exact
// (a < b <=> a - b < 0 for finite, non-overflowing operands; all boundaries and
// clamped args here are small finite values).
__attribute__((always_inline)) sfpi_inline sfpi::vFloat sub(sfpi::vFloat a, sfpi::vFloat b) {
    return __builtin_rvtt_sfpmad(a.get(), sfpi::vFloat(1.0f).get(), (-b).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// Newton-Raphson reciprocal 1/x (RATIONAL denominator). Quasar sfpi has no vFloat
// operator/; use the SFPNONLINEAR recip approximation plus two NR steps
// (y <- y*(2 - x*y)), formed with fused FMA (no SFPADDI). Mirrors rr_recip in the
// tt-llk reference.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat recip(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
#pragma GCC unroll 1
    for (int it = 0; it < 2; it++) {
        const sfpi::vFloat t = fma_const(x, y, -2.0f);  // t = x*y - 2
        y = __builtin_rvtt_sfpmad(
            y.get(), (-t).get(), sfpi::vFloat(0.0f).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);  // y*(2-x*y)
    }
    return y;
}

// =====================================================================
// NEWTON-ROOT evaluator (method 9). STANDALONE: magic-seed + Newton/Householder over
// the full input domain (no segment cascade). Ported verbatim from the proven tt-llk
// reference generic_lut_newton_root_quasar_test.cpp, which is itself a byte-faithful
// replica of the production EVAL_METHOD_NEWTON_ROOT path. All intrinsics
// (reinterpret, >>, vConst1, the int subtract MAGIC - i) exist on Quasar sfpi; the one
// BH-only op (addexp(x,-1) == x*0.5 via SFPDIVP2) is missing on the Quasar ttsim build
// and is substituted with the bit-identical multiply x * 0.5f.
// =====================================================================
#if LUT_RR_METHOD == 9

// addexp(v, -1) == v * 0.5. 0.5 is an exact fp32 value, so x*0.5f is bit-identical to
// the exponent-decrement op for all finite normal inputs (subnormals are out of the
// newton_root domain). Used because SFPADDEXP / SFPDIVP2 is missing on Quasar ttsim.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat nr_half(sfpi::vFloat v) { return v * sfpi::vFloat(0.5f); }

#if (LUT_NR_N == 2) && (LUT_NR_RECIPROCAL == 0)
// --- sqrt: magic seed + double-Newton (native parity) ------------------------
// SFPADDI AVOIDANCE: the tt-llk reference forms `c1 + c*(c2 + c)` and `1 + neg_y*xy`
// with bare vFloat+const adds. On Quasar a standalone `acc + const` lowers to SFPADDI,
// which is UNIMPLEMENTED on the ttsim build (MissingSpecification: tensix_execute_sfpaddi).
// Re-express every `acc + const` as a fused SFPMAD (fma_const / __builtin_rvtt_sfpmad)
// with the constant in the SFPMAD operand slot — numerically identical, never SFPADDI.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat newton_root_eval(sfpi::vFloat x) {
    const sfpi::vInt magic_seed = (sfpi::vInt)(int)(LUT_NR_MAGIC);

    sfpi::vInt i = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(magic_seed - i);

    sfpi::vFloat xy = x * y;
    sfpi::vFloat negative_y = -y;
    sfpi::vFloat c = negative_y * xy;
    // y = y * (c1 + c*(c2 + c)); c2 + c == fma(c, 1, c2); then c1 + c*inner == fma(c, inner, c1).
    const sfpi::vFloat inner = fma_const(c, sfpi::vFloat(1.0f), LUT_NR_C2);
    const sfpi::vFloat poly =
        __builtin_rvtt_sfpmad(c.get(), inner.get(), const_vf(LUT_NR_C1).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    y = y * poly;
    xy = x * y;
    negative_y = -y;
    // one_minus_xyy = 1 + neg_y*xy == fma(neg_y, xy, 1.0) (1.0 in the SFPMAD addend slot, not SFPADDI).
    sfpi::vFloat one_minus_xyy =
        __builtin_rvtt_sfpmad(negative_y.get(), xy.get(), const_vf(1.0f).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    sfpi::vFloat half_xy = nr_half(xy);  // addexp(xy,-1) substitute
    sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
    v_if(sfpi::reinterpret<sfpi::vInt>(x) < sfpi::reinterpret<sfpi::vInt>(infinity)) {
        // y = one_minus_xyy*half_xy + xy (xy is a register addend, a normal 3-reg FMA).
        y = __builtin_rvtt_sfpmad(one_minus_xyy.get(), half_xy.get(), xy.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    v_endif;
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    return y;
}
#elif (LUT_NR_N == 2) && (LUT_NR_RECIPROCAL == 1)
// --- rsqrt: classic inverse-sqrt magic seed + Newton -------------------------
// y = y * (c1 - half_x*(y*y)) re-expressed as a fused SFPMAD (see sqrt note re SFPADDI):
// c1 - half_x*yy == fma((-half_x), yy, c1) with c1 in the SFPMAD operand slot.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat newton_root_eval(sfpi::vFloat x) {
    const sfpi::vInt magic_seed = (sfpi::vInt)(int)(LUT_NR_MAGIC);

    sfpi::vInt i = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(magic_seed - i);
    const sfpi::vFloat neg_half_x = -nr_half(x);  // -0.5*x (addexp(x,-1) substitute, negated)
#pragma GCC unroll 4
    for (int s = 0; s < LUT_NR_ITERS; s++) {
        const sfpi::vFloat yy = y * y;
        const sfpi::vFloat inner = __builtin_rvtt_sfpmad(
            neg_half_x.get(),
            yy.get(),
            const_vf(LUT_NR_C1).get(),
            sfpi::SFPMAD_MOD1_OFFSET_NONE);  // c1 - 0.5*x*y*y
        y = y * inner;
    }
    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    v_if(x == 0.0f) { y = std::numeric_limits<float>::infinity(); }
    v_endif;
    return y;
}
#elif (LUT_NR_N == 3)
// --- cbrt: minimal exponent seed + DIVISION-FREE cubic Householder -----------
__attribute__((always_inline)) sfpi_inline sfpi::vFloat newton_root_eval(sfpi::vFloat x) {
    sfpi::vInt sign_bits = sfpi::reinterpret<sfpi::vInt>(x) & (sfpi::vInt)0x80000000;
    sfpi::vFloat ax = sfpi::setsgn(x, 0);

    sfpi::vInt e_int = sfpi::exexp(ax, sfpi::ExponentMode::NoDebias) - 127;
    sfpi::vFloat m = sfpi::setexp(ax, 127);

    const sfpi::vFloat magic = sfpi::reinterpret<sfpi::vFloat>((sfpi::vInt)(int)0x4B400000);
    // int -> float via vSMag (the only Quasar int->float path).
    sfpi::vFloat ef = sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(e_int), sfpi::RoundMode::Nearest);
    v_if(e_int < 0) {
        ef = -sfpi::convert<sfpi::vFloat>(sfpi::convert<sfpi::vSMag>(~e_int + 1), sfpi::RoundMode::Nearest);
    }
    v_endif;
    sfpi::vInt q = sfpi::reinterpret<sfpi::vInt>(ef * (1.0f / 3.0f) + magic) - sfpi::reinterpret<sfpi::vInt>(magic);

    sfpi::vFloat wm = -0.27f * m + 1.25f;
    sfpi::vFloat w = sfpi::setexp(wm, sfpi::exexp(wm, sfpi::ExponentMode::NoDebias) - q);

    const sfpi::vFloat a13 = 1.0f / 3.0f;
    const sfpi::vFloat a29 = 2.0f / 9.0f;
    for (int s = 0; s < LUT_NR_ITERS; s++) {
        sfpi::vFloat c = 1.0f - ax * (w * w * w);
        w = w * (1.0f + c * a13 + (c * c) * a29);
    }
    sfpi::vFloat y = ax * w * w;
    y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(y) | sign_bits);
    v_if(x == 0.0f) { y = 0.0f; }
    v_endif;
    return y;
}
#endif

#endif  // LUT_RR_METHOD == 9

// =====================================================================
// Range-reduction primitives (exponent / trig family). Ported verbatim from the
// proven tt-llk Quasar generic-LUT reference (generic_lut_activation_quasar_test.cpp):
// reduce/reconstruct math via the Quasar SFPU exexp/setexp/convert ops. All adds that
// the SFPADDI-fold rule could trip are expressed as fused SFPMAD (fma_const / explicit
// __builtin_rvtt_sfpmad). int<->float goes through vSMag (the only int<->float path on
// Quasar sfpi). Compiled away when LUT_RR_METHOD == 0.
// =====================================================================
#if LUT_RR_METHOD != 0

// Reduced-domain mantissa m in [1,2): replace x's biased exponent with 127 (np.frexp
// adjustment m=2*m, e=e-1 -> m in [1,2)). setexp idiom from the BH log kernel.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat rr_mantissa(sfpi::vFloat in) { return sfpi::setexp(in, 127); }

// int -> float (vInt -> vSMag -> vFloat); the only int->float path on Quasar.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat rr_int_to_float(sfpi::vInt v) {
    const auto s = sfpi::convert<sfpi::vSMag>(v);
    return sfpi::convert<sfpi::vFloat>(s, sfpi::RoundMode::Nearest);
}

// True signed exponent e as float (e==0 for x in [1,2)). exexp(Debias) returns the
// frexp-adjusted signed exponent directly (matches the BH log kernel pairing).
__attribute__((always_inline)) sfpi_inline sfpi::vFloat rr_exp_float(sfpi::vFloat in) {
    return rr_int_to_float(sfpi::exexp(in, sfpi::ExponentMode::Debias));
}

// Variable ldexp: result = mant * 2^(e_int). Synthesize 2^e by writing the biased
// exponent field (e+127) onto 1.0 and multiplying. Valid for |e_int| <= 127.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat rr_ldexp(sfpi::vFloat mant, sfpi::vInt e_int) {
    const sfpi::vInt e_biased = e_int + sfpi::vInt(127);
    const sfpi::vFloat two_pow_e = sfpi::setexp(sfpi::vFloat(1.0f), e_biased);
    return mant * two_pow_e;
}

// round-to-nearest float -> vInt via SFPSTOCHRND fp32->sm16 (RNE) then SFPCAST sm32->int.
// Reduced exponent integers (|q|,|i|,|e| <= ~127) fit in 16 bits, so this is exact.
__attribute__((always_inline)) sfpi_inline sfpi::vInt rr_round_to_int(sfpi::vFloat t) {
    const sfpi::vSMag s = sfpi::vSMag(
        __builtin_rvtt_sfpstochrnd_i(t.get(), 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_SMAG16, sfpi::SFPSTOCHRND_RND_EVEN));
    return sfpi::convert<sfpi::vInt>(s);
}

// floor(t) -> vInt. convert<vSMag> only rounds to nearest, so round then correct down
// when the rounded value overshot the input (toward -inf). The fi > t compare is a
// vFloat-vFloat compare; form (fi - t) and compare against literal 0.0f (the same
// nonzero-constant-compare workaround used by sub() elsewhere in this kernel).
__attribute__((always_inline)) sfpi_inline sfpi::vInt rr_floor_to_int(sfpi::vFloat t) {
    sfpi::vInt i = rr_round_to_int(t);
    const sfpi::vFloat fi = rr_int_to_float(i);
    v_if(sub(fi, t) > 0.0f) { i = i - sfpi::vInt(1); }
    v_endif;
    return i;
}

// Floored divmod by compile-time divisor d (>0): e = d*q + r, r in [0,d). |e| <= 127
// is exactly fp32-representable. r is formed in the float domain via fused FMA (no
// SFPADDI: vInt has no multiply on Quasar) then converted back to int.
__attribute__((always_inline)) sfpi_inline void rr_divmod_floor(sfpi::vInt e, int d, sfpi::vInt& q, sfpi::vInt& r) {
    const sfpi::vFloat ef = rr_int_to_float(e);
    q = rr_floor_to_int(ef * sfpi::vFloat(1.0f / (float)d));
    const sfpi::vFloat qf = rr_int_to_float(q);
    const sfpi::vFloat rf =
        __builtin_rvtt_sfpmad(qf.get(), sfpi::vFloat(-(float)d).get(), ef.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    r = rr_round_to_int(rf);
}

// Multiply v by scale-table entry C[r], r in {0,1,2}. vInt == vInt(k) compares; these
// are integer compares (not the float nonzero-constant-compare bug), used as in the
// reference. Top-level v_if (no nesting) per the predicate-stack rule.
__attribute__((always_inline)) sfpi_inline sfpi::vFloat rr_scale_by_r(sfpi::vFloat v, sfpi::vInt r) {
    sfpi::vFloat out = v * sfpi::vFloat(LUT_RR_SCALE0);  // r==0 default
    v_if(r == sfpi::vInt(1)) { out = v * sfpi::vFloat(LUT_RR_SCALE1); }
    v_endif;
    v_if(r == sfpi::vInt(2)) { out = v * sfpi::vFloat(LUT_RR_SCALE2); }
    v_endif;
    return out;
}

// Parity p in {0,1} of a signed integer kv: p = kv - 2*floor(kv/2), in the float
// domain. Returns a vFloat exactly 0.0f (even) or 1.0f (odd).
__attribute__((always_inline)) sfpi_inline sfpi::vFloat rr_parity_f(sfpi::vInt kv) {
    const sfpi::vFloat kf = rr_int_to_float(kv);
    const sfpi::vInt h = rr_floor_to_int(kf * sfpi::vFloat(0.5f));  // floor(k/2)
    const sfpi::vFloat hf = rr_int_to_float(h);
    return __builtin_rvtt_sfpmad(hf.get(), sfpi::vFloat(-2.0f).get(), kf.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

#endif  // LUT_RR_METHOD != 0

// Horner step recursion with COMPILE-TIME coefficient indices (folds LUT_DATA away).
// __attribute__((always_inline)) on the recursive templates prevents LTO
// constprop.isra register spills that otherwise drop the cumulative v_if overrides
// (the metal JIT optimizes differently from the tt-llk standalone build). Applies
// coefficients c_K down to c_0 (BASE points at c_0).
template <std::uint32_t BASE, int K>
__attribute__((always_inline)) sfpi_inline sfpi::vFloat horner_step(sfpi::vFloat acc, sfpi::vFloat x) {
    if constexpr (K < 0) {
        return acc;
    } else {
        acc = horner_fma(acc, x, LUT_DATA[BASE + static_cast<std::uint32_t>(K)]);
        return horner_step<BASE, K - 1>(acc, x);
    }
}

// Horner-eval a degree-DEG polynomial whose coefficients c0..cDEG live at
// LUT_DATA[BASE .. BASE+DEG]. acc starts at c_DEG, then folds down.
template <std::uint32_t BASE, std::uint32_t DEG>
__attribute__((always_inline)) sfpi_inline sfpi::vFloat horner_poly(sfpi::vFloat x) {
    const sfpi::vFloat acc = const_vf(LUT_DATA[BASE + DEG]);  // leading coeff (may be -1.0): bit-load, not CREG
    return horner_step<BASE, static_cast<int>(DEG) - 1>(acc, x);
}

// Clamp x to segment SEG's sub-interval [b_SEG, b_{SEG+1}] (top-level v_if,
// compare-against-zero form) so no polynomial is evaluated outside its fit range.
template <std::uint32_t SEG>
__attribute__((always_inline)) sfpi_inline sfpi::vFloat clamp_to_segment(sfpi::vFloat x_clamped) {
    const sfpi::vFloat seg_lo = const_vf(LUT_DATA[SEG]);
    const sfpi::vFloat seg_hi = const_vf(LUT_DATA[SEG + 1]);
    sfpi::vFloat xs = x_clamped;
    v_if(sub(xs, seg_lo) < 0.0f) { xs = seg_lo; }
    v_endif;
    v_if(sub(xs, seg_hi) > 0.0f) { xs = seg_hi; }
    v_endif;
    return xs;
}

// eval_seg<SEG>: evaluate segment SEG (POLY or RATIONAL) on the clamped argument.
template <std::uint32_t SEG>
__attribute__((always_inline)) sfpi_inline sfpi::vFloat eval_seg(sfpi::vFloat x_clamped) {
    constexpr std::uint32_t base = COEFF_OFFSET + SEG * COEFFS_PER_SEGMENT;
    const sfpi::vFloat xs = clamp_to_segment<SEG>(x_clamped);
    if constexpr (EVAL_METHOD == 1) {
        // RATIONAL: P(x)/Q(x). Numerator at [base .. base+NUM_DEGREE],
        // denominator at [base+NUM_DEGREE+1 .. base+NUM_DEGREE+1+DEN_DEGREE].
        constexpr std::uint32_t num_base = base;
        constexpr std::uint32_t den_base = base + (NUM_DEGREE + 1);
        const sfpi::vFloat p = horner_poly<num_base, NUM_DEGREE>(xs);
        const sfpi::vFloat q = horner_poly<den_base, DEN_DEGREE>(xs);
        return p * recip(q);
    } else {
        return horner_poly<base, POLY_DEGREE>(xs);
    }
}

// Cumulative segment-override chain via template recursion over compile-time SEG.
// Each segment's (already-bounded) value is computed BEFORE the select v_if so its
// clamp v_ifs stay at top level (a nested v_if corrupts the Quasar predicate stack).
// The select predicate uses the compare-against-zero form.
template <std::uint32_t SEG>
__attribute__((always_inline)) sfpi_inline void select_segment(sfpi::vFloat& result, sfpi::vFloat x_clamped) {
    if constexpr (SEG < NUM_SEGMENTS) {
        const sfpi::vFloat b_seg = const_vf(LUT_DATA[SEG]);
        const sfpi::vFloat segval = eval_seg<SEG>(x_clamped);
        v_if(sub(x_clamped, b_seg) >= 0.0f) { result = segval; }
        v_endif;
        select_segment<SEG + 1>(result, x_clamped);
    }
}

// Evaluate the embedded piecewise LUT on the current Dest row(s).
//
// With range reduction (LUT_RR_METHOD != 0): REDUCE x_in to the polynomial argument
// r_arg (+ per-method reconstruct state) BEFORE the poly cascade, then RECONSTRUCT the
// full activation over the original domain AFTER. Ported verbatim from the proven
// tt-llk Quasar generic-LUT reference. With LUT_RR_METHOD == 0 (no RR) r_arg is x_in
// and both blocks compile away -> byte-identical to the no-RR path.
sfpi_inline void piecewise_generic_lut_row() {
    const sfpi::vFloat x_in = sfpi::dst_reg[0];

#if LUT_RR_METHOD == 9
    // NEWTON-ROOT (method 9) is a STANDALONE evaluator over the full input domain: the
    // magic seed folds in the exponent decomposition, so there is no segment cascade,
    // no clamp, and no reduce/reconstruct. Evaluate and store directly.
    sfpi::dst_reg[0] = newton_root_eval(x_in);
    return;
#else

    // ---- REDUCE: r_arg = reduced polynomial argument; capture reconstruct state.
    sfpi::vFloat r_arg = x_in;
#if LUT_RR_METHOD == 1  // log: x = 2^e * m
    const sfpi::vFloat rr_ef = rr_exp_float(x_in);
    r_arg = rr_mantissa(x_in);
#elif LUT_RR_METHOD == 2  // exp (Cody-Waite): q=round(x*mult); s = x - q*const
    const sfpi::vFloat t_q = x_in * const_vf(LUT_RR_EXP_MULT);
    const sfpi::vInt q_i = rr_round_to_int(t_q);
    const sfpi::vFloat q_f = rr_int_to_float(q_i);
    r_arg = __builtin_rvtt_sfpmad(
        q_f.get(), const_vf(-(float)(LUT_RR_EXP_CONST)).get(), x_in.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
#elif LUT_RR_METHOD == 3  // cbrt: |x| = 2^e * m, e = 3q + r
    const sfpi::vFloat ax = sfpi::abs(x_in);
    const sfpi::vInt e_i = sfpi::exexp(ax, sfpi::ExponentMode::Debias);
    sfpi::vInt q_i, r_i;
    rr_divmod_floor(e_i, 3, q_i, r_i);
    r_arg = rr_mantissa(ax);
#elif LUT_RR_METHOD == 4  // expalu_exp2: t = x*mult; i = floor(t); f = t - i
    const sfpi::vFloat t_e = x_in * const_vf(LUT_RR_EXP2_MULT);
    const sfpi::vInt i_i = rr_floor_to_int(t_e);
    const sfpi::vFloat i_f = rr_int_to_float(i_i);
    r_arg =
        __builtin_rvtt_sfpmad(const_vf(-1.0f).get(), i_f.get(), t_e.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);  // f = t - i
#elif LUT_RR_METHOD == 5  // expalu_log2: (x+off) = 2^e * m
    const sfpi::vFloat xo = x_in + const_vf(LUT_RR_INPUT_OFFSET);
    const sfpi::vFloat rr_ef = rr_exp_float(xo);
    const sfpi::vFloat m_v = rr_mantissa(xo);
#if LUT_RR_LOG2_BASIS_MMINUS1
    r_arg = m_v - const_vf(1.0f);  // Horner on u = m-1 (c0 == 0)
#else
    r_arg = m_v;
#endif
#elif LUT_RR_METHOD == 6  // expalu_pow: |x| = 2^e * m, e = n*q + r
    const sfpi::vFloat ax = sfpi::abs(x_in);
    const sfpi::vInt e_i = sfpi::exexp(ax, sfpi::ExponentMode::Debias);
    sfpi::vInt q_i, r_i;
    rr_divmod_floor(e_i, LUT_RR_POW_N, q_i, r_i);
    r_arg = rr_mantissa(ax);
#elif LUT_RR_METHOD == 7  // trig (sin/cos): k=round(x/pi); s = x - k*pi
    const sfpi::vFloat t_k = x_in * const_vf(LUT_RR_TRIG_INV_PI);
    const sfpi::vInt k_i = rr_round_to_int(t_k);
    const sfpi::vFloat k_f = rr_int_to_float(k_i);
    r_arg = __builtin_rvtt_sfpmad(
        k_f.get(), const_vf(-(float)(LUT_RR_TRIG_PI)).get(), x_in.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    const sfpi::vFloat trig_sign = rr_parity_f(k_i);  // 0.0 (even k) or 1.0 (odd k)
#elif LUT_RR_METHOD == 8  // tan: j=round(x/(pi/2)); Cody-Waite a = x - j*(pi/2)
    const sfpi::vFloat t_j = x_in * const_vf(LUT_RR_TAN_INV_HALFPI);
    const sfpi::vInt j_i = rr_round_to_int(t_j);
    const sfpi::vFloat j_f = rr_int_to_float(j_i);
    const sfpi::vFloat a_hi = __builtin_rvtt_sfpmad(
        j_f.get(), const_vf(-(float)(LUT_RR_TAN_PI2_HI)).get(), x_in.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    r_arg = __builtin_rvtt_sfpmad(
        j_f.get(), const_vf(-(float)(LUT_RR_TAN_PI2_LO)).get(), a_hi.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    const sfpi::vFloat tan_odd = rr_parity_f(j_i);  // 0.0 (even j) or 1.0 (odd j)
#endif

    // Clamp r_arg to [b0, bN] (the REDUCED domain), compare-against-zero form.
    const sfpi::vFloat b_lo = const_vf(LUT_DATA[0]);
    const sfpi::vFloat b_hi = const_vf(LUT_DATA[NUM_SEGMENTS]);
    sfpi::vFloat x_clamped = r_arg;
    v_if(sub(x_clamped, b_lo) < 0.0f) { x_clamped = b_lo; }
    v_endif;
    v_if(sub(x_clamped, b_hi) > 0.0f) { x_clamped = b_hi; }
    v_endif;

    sfpi::vFloat result = eval_seg<0>(x_clamped);
    select_segment<1>(result, x_clamped);

    // ---- RECONSTRUCT the full activation over the original domain.
#if LUT_RR_METHOD == 1  // log: e*ln2 + P(m)
    result =
        __builtin_rvtt_sfpmad(rr_ef.get(), const_vf(LUT_RR_LOG_LN2).get(), result.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    v_if(x_in == sfpi::vFloat(0.0f)) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;
#elif LUT_RR_METHOD == 2   // exp: 2^q * P(s)
    result = rr_ldexp(result, q_i);
#elif LUT_RR_METHOD == 3   // cbrt: sign * C[r] * 2^q * P(m)
    result = rr_scale_by_r(result, r_i);
    result = rr_ldexp(result, q_i);
    result = sfpi::copysgn(result, x_in);
    v_if(sfpi::abs(x_in) == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(0.0f); }
    v_endif;
#elif LUT_RR_METHOD == 4   // expalu_exp2: 2^i * P(f) [+ compose]
    result = rr_ldexp(result, i_i);
#if LUT_RR_COMPOSE == 1    // sigmoid: 1/(1+y)
    result = recip(result + const_vf(1.0f));
#elif LUT_RR_COMPOSE == 2  // minus_one: y - 1
    result = result - const_vf(1.0f);
#endif
#elif LUT_RR_METHOD == 5  // expalu_log2: scale*(e + P(u))
    result = (rr_ef + result) * const_vf(LUT_RR_LOG2_SCALE);
#elif LUT_RR_METHOD == 6  // expalu_pow: [1/](sign) * C[r] * 2^q * P(m)
    result = rr_scale_by_r(result, r_i);
    result = rr_ldexp(result, q_i);
#if (LUT_RR_POW_N % 2) == 1
    result = sfpi::copysgn(result, x_in);  // odd root: restore sign
#endif
#if LUT_RR_POW_RECIP
    result = recip(result);  // rsqrt
#endif
    v_if(sfpi::abs(x_in) == sfpi::vFloat(0.0f)) {
#if LUT_RR_POW_RECIP
        result = std::numeric_limits<float>::infinity();
#else
        result = sfpi::vFloat(0.0f);
#endif
    }
    v_endif;
#elif LUT_RR_METHOD == 7  // trig: result = (-1)^k * P(s); odd-k parity flips sign.
    // trig_sign is exactly 0.0f or 1.0f; compare against 1.0f via the sub() workaround.
    v_if(sub(trig_sign, const_vf(1.0f)) == 0.0f) { result = -result; }
    v_endif;
#elif LUT_RR_METHOD == 8  // tan: j even -> P(a); j odd -> -1/P(a)
    const sfpi::vFloat tan_recip = -recip(result);  // -1/P(a), computed on all lanes
    v_if(sub(tan_odd, const_vf(1.0f)) == 0.0f) { result = tan_recip; }
    v_endif;
#endif

    sfpi::dst_reg[0] = result;
#endif  // LUT_RR_METHOD == 9 (standalone) else (cascade)
}

}  // namespace lut_detail

// SFPU entry point invoked by _llk_math_eltwise_unary_sfpu_params_. Iterates over
// the SFPU's Dest window. On Quasar SFP_ROWS==2 per iteration; on WH/BH the legacy
// 0..ITERATIONS dst_reg loop is used.
template <int ITERATIONS = 8>
inline void calculate_lut_activation() {
    // Program CREG 11 = -1.0 (craq-sim leaves it uninitialized; see init_neg1_const_reg).
    lut_detail::init_neg1_const_reg();
#ifdef ARCH_QUASAR
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        lut_detail::piecewise_generic_lut_row();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // dest_reg += 2 rows
    }
#else
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        lut_detail::piecewise_generic_lut_row();
        sfpi::dst_reg++;
    }
#endif
}

}  // namespace sfpu
}  // namespace ckernel
