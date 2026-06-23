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
sfpi_inline sfpi::vFloat fma_const(sfpi::vFloat a, sfpi::vFloat b, float c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), const_vf(c).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// Horner FMA step: acc*x + c. Passes the runtime register `x` as the SFPMAD multiplicand.
sfpi_inline sfpi::vFloat horner_fma(sfpi::vFloat acc, sfpi::vFloat x, float c) {
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
sfpi_inline sfpi::vFloat sub(sfpi::vFloat a, sfpi::vFloat b) {
    return __builtin_rvtt_sfpmad(a.get(), sfpi::vFloat(1.0f).get(), (-b).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

// Newton-Raphson reciprocal 1/x (RATIONAL denominator). Quasar sfpi has no vFloat
// operator/; use the SFPNONLINEAR recip approximation plus two NR steps
// (y <- y*(2 - x*y)), formed with fused FMA (no SFPADDI). Mirrors rr_recip in the
// tt-llk reference.
sfpi_inline sfpi::vFloat recip(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
#pragma GCC unroll 1
    for (int it = 0; it < 2; it++) {
        const sfpi::vFloat t = fma_const(x, y, -2.0f);  // t = x*y - 2
        y = __builtin_rvtt_sfpmad(
            y.get(), (-t).get(), sfpi::vFloat(0.0f).get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);  // y*(2-x*y)
    }
    return y;
}

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
sfpi_inline void piecewise_generic_lut_row() {
    const sfpi::vFloat x_in = sfpi::dst_reg[0];

    // Clamp x to [b0, bN] (NO range reduction in this slice), compare-against-zero form.
    const sfpi::vFloat b_lo = const_vf(LUT_DATA[0]);
    const sfpi::vFloat b_hi = const_vf(LUT_DATA[NUM_SEGMENTS]);
    sfpi::vFloat x_clamped = x_in;
    v_if(sub(x_clamped, b_lo) < 0.0f) { x_clamped = b_lo; }
    v_endif;
    v_if(sub(x_clamped, b_hi) > 0.0f) { x_clamped = b_hi; }
    v_endif;

    sfpi::vFloat result = eval_seg<0>(x_clamped);
    select_segment<1>(result, x_clamped);

    sfpi::dst_reg[0] = result;
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
