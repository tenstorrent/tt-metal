// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench of genify_matmul's swiglu epilogue: one fronted Float32 accumulator block
// (block_m x block_n tiles, TileRowMajor, gate half then up half per row -- or natural (gate, up)
// pairs) -> block_m x block_n/2 output tiles, out = silu(gate) * up.
//
// Every variant is an eltwise_chain (the op's helper); candidates differ only in the SFPU element:
//   0 baseline   CopyTile g->D0, Silu (silu_tile<fp32_prec>), CopyTile u->D1, MulBinary, PackTile
//                (exactly the op's run_epilogue swiglu branch)
//   1 fused      CopyTile g->D0, CopyTile u->D1, SiluMul<impl 0>: the SAME sigmoid math as silu_tile
//                (exp_21f or Juffa fp32 exp + Newton reciprocal) fused with the * up in one SFPU
//                pass (no MulBinary pass, no intermediate bf16 rounding of silu)
//   2 fast3      SiluMul<impl 1>: sigmoid(|g|) = 1/(1+2^(-|g| log2e)), 2^t by magic-number round +
//                degree-3 2^f polynomial + exponent add, Newton reciprocal on the normalized (1,2]
//                denominator (no NaN guard needed), sigmoid(g<0) = e * sigmoid(|g|)
//   3 fast2      as 2 with a degree-2 polynomial
//   4 fast3_nr0  as 2 with no Newton step (SFPARECIP 7-bit only) -- precision-cost option
//   6..9         unary-addressed twins of 1..4 (fusedU, fast3U, fast2U, fast3_nr0U): DEST base at the
//                gate tile, up/out at compile-time offsets (see calculate_swiglu_same_u)
//   10, 11       sym3U / sym2U: branch-free symmetric formulation (calculate_swiglu_sym_u)
//   12, 13       symh1U / symh2U: sym3 with fp16-exact k2/k3 literals, 1 or 2 interleaved rows
//   14           spU: symh math software-pipelined across rows (calculate_swiglu_sp_u)
//   15, 16       spiU / spi2U: statement-interleaved pipeline (deg 3 / deg 2)
//   17           spikU: spiU with k3 hoisted in an LReg
//   18 (spikkU)  does not compile (too few lregs) -- kept only as a record
//   19 spi2kU    spi2U with k2 hoisted
//   20 spi4U     fp32-grade: degree-4 2^f + two Newton steps, interleaved
//   21 grad      kernels/swiglu_glu_sfpu.hpp SwiGluMul<silu_fp32_precision> (spikU / spi4U, packaged)
//   5 floor      CopyTile g->D0, CopyTile u->D1, PackTile (no SFPU) -- unpack/copy/pack floor
// None of these bypass the chain: SiluMul is a chain element through the documented BinaryOp CRTP
// extension point (like the op's local Silu); its SFPU body is raw sfpi (the SFPU function itself
// is new, there is no library function computing silu(a)*b).

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"  // MulBinary
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "swiglu_glu_sfpu.hpp"  // the graduation candidate (variant 21)

#ifdef TRISC_MATH
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

// impl 0: the op's silu_tile sigmoid math, fused with * up. One rounding (the packer's / the final
// bf16 convert on a 16-bit DEST) instead of silu_tile<false>'s bf16 round + MulBinary's.
template <bool fp32_sigmoid, bool dest16, int ITERATIONS = 8>
inline void calculate_swiglu_same(const uint g_idx, const uint u_idx, const uint o_idx) {
    constexpr uint ts = 32;
    HoistedIf<!fp32_sigmoid> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    HoistedIf<fp32_sigmoid> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0;
    constexpr float p1 = EXP_FP32_P1;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[g_idx * ts];
        sfpi::vFloat r = g * _sfpu_sigmoid_prgm_<fp32_sigmoid>(g, c0, c1, c2, neg_ln2_hi, p0, p1);
        sfpi::vFloat u = sfpi::dst_reg[u_idx * ts];
        r = r * u;
        if constexpr (dest16) {
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[o_idx * ts] = r;
        sfpi::dst_reg++;
    }
}

inline void swiglu_same_init() { sigmoid_init<false>(); }  // Prgm0 = 2.0, Prgm1 = 1/ln2

// impl 1..3 constants (Prgm0 = 2.0 Newton, Prgm1 = -log2(e), Prgm2 = 1.5 * 2^23 rounding magic).
constexpr float SWG_NEG_LOG2E = -1.4426950408889634f;
constexpr float SWG_MAGIC = 12582912.0f;  // 1.5 * 2^23: t + M rounds t to an integer in M's low bits
constexpr float SWG_ARG_MAX = 87.0f;      // |g| clamp: 2^(-87 log2e) stays a normal fp32
// 2^f on [-0.5, 0.5], relative-minimax (fit host-side):
//   degree 3 (c0 = 1, so the last Horner step adds vConst1) max rel err 2^-13.0; degree 2 (free c0) 2^-9.0
constexpr float SWG_P3_1 = 0.6932254433631897f, SWG_P3_2 = 0.24229750037193298f, SWG_P3_3 = 0.05533519387245178f;
constexpr float SWG_P2_0 = 1.0004717111587524f, SWG_P2_1 = 0.7041206955909729f, SWG_P2_2 = 0.23876285552978516f;

template <int DEG, int NEWTON, bool dest16, int ITERATIONS = 8>
inline void calculate_swiglu_fast(const uint g_idx, const uint u_idx, const uint o_idx) {
    constexpr uint ts = 32;
    // Hoisted polynomial coefficients (sfpi never lifts an fp32 literal out of a loop by itself).
    sfpi::vFloat k3 = DEG == 3 ? SWG_P3_3 : SWG_P2_2;
    sfpi::vFloat k2 = DEG == 3 ? SWG_P3_2 : SWG_P2_1;
    sfpi::vFloat k1 = DEG == 3 ? SWG_P3_1 : SWG_P2_0;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[g_idx * ts];
        sfpi::vFloat a = sfpi::min(sfpi::abs(g), SWG_ARG_MAX);
        // t = -|g| log2e;  z = t + M (n = round(t) in z's low mantissa bits);  f = t - n in [-.5, .5]
        sfpi::vFloat z = a * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;
        sfpi::vFloat nn = sfpi::vConstFloatPrgm2 - z;  // = -n, exact
        sfpi::vFloat f = a * sfpi::vConstFloatPrgm1 + nn;
        sfpi::vFloat p;
        if constexpr (DEG == 3) {
            p = (k3 * f + k2) * f + k1;
            p = p * f + 1.0f;
        } else {
            p = (k3 * f + k2) * f + k1;
        }
        // e = p * 2^n: bits(z) << 23 == n << 23 (mod 2^32) since bits(M) = 0x4B400000.
        sfpi::vFloat e = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + (sfpi::as<sfpi::vInt>(z) << 23));
        sfpi::vFloat den = e + 1.0f;  // in (1, 2]
        sfpi::vFloat r = sfpi::approx_recip(den);
        if constexpr (NEWTON > 0) {
            sfpi::vFloat t = den * r - sfpi::vConstFloatPrgm0;
            r = r * -t - 0.0f;
        }
        // sigmoid(g) = r for g >= 0, e * r for g < 0
        v_if(g < 0.0f) { r = r * e; }
        v_endif;
        sfpi::vFloat u = sfpi::dst_reg[u_idx * ts];
        r = g * r * u;
        if constexpr (dest16) {
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[o_idx * ts] = r;
        sfpi::dst_reg++;
    }
}

// Unary-addressed twins: _llk_math_eltwise_unary_sfpu_params_ sets the DEST base to the gate tile, and
// the up / out tiles are COMPILE-TIME row offsets (UOFF / OOFF = slot distance * 32). The binary wrapper
// (SFPU_BINARY_CALL, used by MulBinary and clamped_silu_glu) starts at DEST 0 and passes the tile
// indices at runtime, so every SFPLOAD / SFPSTORE address is runtime and the compiler must build those
// instruction words on the RISC-V instead of issuing static TTI words / replaying the loop.
template <bool fp32_sigmoid, bool dest16, int UOFF, int OOFF, int ITERATIONS = 8>
inline void calculate_swiglu_same_u() {
    HoistedIf<!fp32_sigmoid> c0 = EXP_21F_C0, c1 = EXP_21F_C1, c2 = EXP_21F_C2;
    HoistedIf<fp32_sigmoid> neg_ln2_hi = EXP_FP32_NEG_LN2_HI, p0 = EXP_FP32_P0;
    constexpr float p1 = EXP_FP32_P1;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[0];
        sfpi::vFloat r = g * _sfpu_sigmoid_prgm_<fp32_sigmoid>(g, c0, c1, c2, neg_ln2_hi, p0, p1);
        sfpi::vFloat u = sfpi::dst_reg[UOFF];
        r = r * u;
        if constexpr (dest16) {
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[OOFF] = r;
        sfpi::dst_reg++;
    }
}

template <int DEG, int NEWTON, bool dest16, int UOFF, int OOFF, int ITERATIONS = 8>
inline void calculate_swiglu_fast_u() {
    sfpi::vFloat k3 = DEG == 3 ? SWG_P3_3 : SWG_P2_2;
    sfpi::vFloat k2 = DEG == 3 ? SWG_P3_2 : SWG_P2_1;
    sfpi::vFloat k1 = DEG == 3 ? SWG_P3_1 : SWG_P2_0;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::min(sfpi::abs(g), SWG_ARG_MAX);
        sfpi::vFloat z = a * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;
        sfpi::vFloat nn = sfpi::vConstFloatPrgm2 - z;
        sfpi::vFloat f = a * sfpi::vConstFloatPrgm1 + nn;
        sfpi::vFloat p = (k3 * f + k2) * f + k1;
        if constexpr (DEG == 3) {
            p = p * f + 1.0f;
        }
        sfpi::vFloat e = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + (sfpi::as<sfpi::vInt>(z) << 23));
        sfpi::vFloat den = e + 1.0f;
        sfpi::vFloat r = sfpi::approx_recip(den);
        if constexpr (NEWTON > 0) {
            sfpi::vFloat t = den * r - sfpi::vConstFloatPrgm0;
            r = r * -t - 0.0f;
        }
        v_if(g < 0.0f) { r = r * e; }
        v_endif;
        sfpi::vFloat u = sfpi::dst_reg[UOFF];
        r = g * r * u;
        if constexpr (dest16) {
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[OOFF] = r;
        sfpi::dst_reg++;
    }
}

// "sym": sigmoid(g) = 1 / (1 + E), E = 2^(-g log2e) on g clamped to [-87, 87] (E in [2^-126, 2^126):
// normal, and its reciprocal never hits SFPARECIP's 0 / inf cases) -> no |g|, no v_if, no sign select.
// The Newton step is written r1 = r0 + r0 * (1 - den * r0) and folded into the output product:
// out = q + q * eps with q = (g * u) * r0, eps = 1 - den * r0 (same precision as y * (2 - x y)), which
// frees Prgm0 for a polynomial coefficient. Constants: Prgm0 = k3 (deg 3) / k2 (deg 2), Prgm1 = -log2e,
// Prgm2 = 1.5 * 2^23.
template <int DEG, bool dest16, int UOFF, int OOFF, int ITERATIONS = 8>
inline void calculate_swiglu_sym_u() {
    sfpi::vFloat k2 = DEG == 3 ? SWG_P3_2 : SWG_P2_1;
    sfpi::vFloat k1 = DEG == 3 ? SWG_P3_1 : SWG_P2_0;
    sfpi::vFloat hi = SWG_ARG_MAX, lo = -SWG_ARG_MAX;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[0];
        sfpi::vFloat u = sfpi::dst_reg[UOFF];
        sfpi::vFloat w = g * u;
        sfpi::vFloat x = sfpi::min(sfpi::max(g, lo), hi);
        sfpi::vFloat z = x * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;  // n = round(-x log2e) in low bits
        sfpi::vFloat nn = sfpi::vConstFloatPrgm2 - z;                          // -n, exact
        sfpi::vFloat f = x * sfpi::vConstFloatPrgm1 + nn;                      // in [-0.5, 0.5]
        sfpi::vFloat p = (sfpi::vConstFloatPrgm0 * f + k2) * f + k1;
        if constexpr (DEG == 3) {
            p = p * f + 1.0f;
        }
        sfpi::vFloat e = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + (sfpi::as<sfpi::vInt>(z) << 23));
        sfpi::vFloat den = e + 1.0f;
        sfpi::vFloat r = sfpi::approx_recip(den);
        sfpi::vFloat eps = 1.0f - den * r;
        sfpi::vFloat q = w * r;
        sfpi::vFloat o = q * eps + q;
        if constexpr (dest16) {
            o = sfpi::convert<sfpi::vFloat16b>(o, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[OOFF] = o;
        sfpi::dst_reg++;
    }
}

template <int DEG>
inline void swiglu_sym_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = DEG == 3 ? SWG_P3_3 : SWG_P2_2;
    sfpi::vConstFloatPrgm1 = SWG_NEG_LOG2E;
    sfpi::vConstFloatPrgm2 = SWG_MAGIC;
}

// "symh": the sym formulation with k2 / k3 exactly fp16-representable (one SFPLOADI each, so they need
// not be hoisted) and k1 in Prgm0 (refit: max rel err 2^-13.1). ROWS = 2 interleaves two independent
// SFPU rows per loop iteration so each row's dependent SFPMAD chain fills the other's latency shadows.
// degree 2 with k0 = 1 (const reg) and fp16-exact k2, k1 in Prgm0: max rel err 2^-8.99
constexpr float SWG_Q_K1 = 0.7029957175254822f, SWG_Q_K2 = 0.239990234375f;
constexpr float SWG_H_K1 = 0.6932584643363953f, SWG_H_K2 = 0.2423095703125f, SWG_H_K3 = 0.055328369140625f;

template <bool dest16, int UOFF, int OOFF, int ROWS, int ITERATIONS = 8>
inline void calculate_swiglu_symh_u() {
    if constexpr (ROWS == 1) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat g = sfpi::dst_reg[0];
            sfpi::vFloat u = sfpi::dst_reg[UOFF];
            sfpi::vFloat w = g * u;
            sfpi::vFloat x = sfpi::min(sfpi::max(g, -SWG_ARG_MAX), SWG_ARG_MAX);
            sfpi::vFloat z = x * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;
            sfpi::vFloat nn = sfpi::vConstFloatPrgm2 - z;
            sfpi::vFloat f = x * sfpi::vConstFloatPrgm1 + nn;
            sfpi::vFloat p = (SWG_H_K3 * f + SWG_H_K2) * f + sfpi::vConstFloatPrgm0;
            p = p * f + 1.0f;
            sfpi::vFloat e = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + (sfpi::as<sfpi::vInt>(z) << 23));
            sfpi::vFloat den = e + 1.0f;
            sfpi::vFloat r = sfpi::approx_recip(den);
            sfpi::vFloat eps = 1.0f - den * r;
            sfpi::vFloat q = w * r;
            sfpi::vFloat o = q * eps + q;
            if constexpr (dest16) {
                o = sfpi::convert<sfpi::vFloat16b>(o, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[OOFF] = o;
            sfpi::dst_reg++;
        }
    } else {
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat ga = sfpi::dst_reg[0];
            sfpi::vFloat gb = sfpi::dst_reg[1];
            sfpi::vFloat xa = sfpi::min(sfpi::max(ga, -SWG_ARG_MAX), SWG_ARG_MAX);
            sfpi::vFloat xb = sfpi::min(sfpi::max(gb, -SWG_ARG_MAX), SWG_ARG_MAX);
            sfpi::vFloat za = xa * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;
            sfpi::vFloat zb = xb * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;
            sfpi::vFloat fa = xa * sfpi::vConstFloatPrgm1 + (sfpi::vConstFloatPrgm2 - za);
            sfpi::vFloat fb = xb * sfpi::vConstFloatPrgm1 + (sfpi::vConstFloatPrgm2 - zb);
            sfpi::vFloat pa = SWG_H_K3 * fa + SWG_H_K2;
            sfpi::vFloat pb = SWG_H_K3 * fb + SWG_H_K2;
            pa = pa * fa + sfpi::vConstFloatPrgm0;
            pb = pb * fb + sfpi::vConstFloatPrgm0;
            pa = pa * fa + 1.0f;
            pb = pb * fb + 1.0f;
            sfpi::vFloat da =
                sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(pa) + (sfpi::as<sfpi::vInt>(za) << 23)) + 1.0f;
            sfpi::vFloat db =
                sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(pb) + (sfpi::as<sfpi::vInt>(zb) << 23)) + 1.0f;
            sfpi::vFloat ra = sfpi::approx_recip(da);
            sfpi::vFloat rb = sfpi::approx_recip(db);
            sfpi::vFloat ea = 1.0f - da * ra;
            sfpi::vFloat eb = 1.0f - db * rb;
            ra = ra * ga;
            rb = rb * gb;
            ra = ra * sfpi::dst_reg[UOFF];
            rb = rb * sfpi::dst_reg[UOFF + 1];
            sfpi::vFloat oa = ra * ea + ra;
            sfpi::vFloat ob = rb * eb + rb;
            if constexpr (dest16) {
                oa = sfpi::convert<sfpi::vFloat16b>(oa, sfpi::RoundMode::Nearest);
                ob = sfpi::convert<sfpi::vFloat16b>(ob, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[OOFF] = oa;
            sfpi::dst_reg[OOFF + 1] = ob;
            sfpi::dst_reg += 2;
        }
    }
}

inline void swiglu_symh_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = SWG_H_K1;
    sfpi::vConstFloatPrgm1 = SWG_NEG_LOG2E;
    sfpi::vConstFloatPrgm2 = SWG_MAGIC;
}

// "sp": symh math, software-pipelined across the face's 8 rows: the HEAD of row i+1 (loads, g*u, clamp,
// z, f) is interleaved with the TAIL of row i (polynomial, exponent add, reciprocal, Newton, store), so
// the two independent dependency chains fill each other's latency shadows. Fully unrolled with static
// row offsets (no replay: every row differs in its DEST offsets).
#define SWG_HEAD(I, W, Z, F)                                                   \
    {                                                                          \
        sfpi::vFloat g_ = sfpi::dst_reg[I];                                    \
        W = g_ * sfpi::dst_reg[UOFF + I];                                      \
        sfpi::vFloat x_ = sfpi::min(sfpi::max(g_, -SWG_ARG_MAX), SWG_ARG_MAX); \
        Z = x_ * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;              \
        F = x_ * sfpi::vConstFloatPrgm1 + (sfpi::vConstFloatPrgm2 - Z);        \
    }

template <bool dest16, int UOFF, int OOFF, int I, int DEG = 3>
sfpi_inline void swg_tail(sfpi::vFloat w, sfpi::vFloat z, sfpi::vFloat f) {
    sfpi::vFloat p;
    if constexpr (DEG == 3) {
        p = (SWG_H_K3 * f + SWG_H_K2) * f + sfpi::vConstFloatPrgm0;
        p = p * f + 1.0f;
    } else {
        p = (SWG_Q_K2 * f + sfpi::vConstFloatPrgm0) * f + 1.0f;
    }
    sfpi::vFloat den = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + (sfpi::as<sfpi::vInt>(z) << 23)) + 1.0f;
    sfpi::vFloat r = sfpi::approx_recip(den);
    sfpi::vFloat eps = 1.0f - den * r;
    sfpi::vFloat q = w * r;
    sfpi::vFloat o = q * eps + q;
    if constexpr (dest16) {
        o = sfpi::convert<sfpi::vFloat16b>(o, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[OOFF + I] = o;
}

template <bool dest16, int UOFF, int OOFF, int ITERATIONS = 8>
inline void calculate_swiglu_sp_u() {
    static_assert(ITERATIONS == 8);
    sfpi::vFloat wa, za, fa, wb, zb, fb;
    SWG_HEAD(0, wa, za, fa);
    SWG_HEAD(1, wb, zb, fb);
    swg_tail<dest16, UOFF, OOFF, 0>(wa, za, fa);
    SWG_HEAD(2, wa, za, fa);
    swg_tail<dest16, UOFF, OOFF, 1>(wb, zb, fb);
    SWG_HEAD(3, wb, zb, fb);
    swg_tail<dest16, UOFF, OOFF, 2>(wa, za, fa);
    SWG_HEAD(4, wa, za, fa);
    swg_tail<dest16, UOFF, OOFF, 3>(wb, zb, fb);
    SWG_HEAD(5, wb, zb, fb);
    swg_tail<dest16, UOFF, OOFF, 4>(wa, za, fa);
    SWG_HEAD(6, wa, za, fa);
    swg_tail<dest16, UOFF, OOFF, 5>(wb, zb, fb);
    SWG_HEAD(7, wb, zb, fb);
    swg_tail<dest16, UOFF, OOFF, 6>(wa, za, fa);
    swg_tail<dest16, UOFF, OOFF, 7>(wb, zb, fb);
    sfpi::dst_reg += 3;  // += 8 in increments the encoding accepts
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 2;
}

// "spi": like "sp" but interleaved at statement level (sfpi keeps source order; it does not schedule
// the tail of row I against the head of row I+1 by itself). STEP: tail(I) on (WA, ZA, FA) alternated
// with head(I+1) into (WB, ZB, FB).
#define SWG_STEP(I, WA, ZA, FA, WB, ZB, FB)                                               \
    {                                                                                     \
        sfpi::vFloat p_ = K3V * FA + K2V;                                                 \
        sfpi::vFloat g_ = sfpi::dst_reg[(I) + 1];                                         \
        p_ = p_ * FA + sfpi::vConstFloatPrgm0;                                            \
        WB = sfpi::dst_reg[UOFF + (I) + 1];                                               \
        p_ = p_ * FA + 1.0f;                                                              \
        WB = WB * g_;                                                                     \
        ZA = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(ZA) << 23);                      \
        g_ = sfpi::max(g_, -SWG_ARG_MAX);                                                 \
        p_ = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p_) + sfpi::as<sfpi::vInt>(ZA)); \
        g_ = sfpi::min(g_, SWG_ARG_MAX);                                                  \
        p_ = p_ + 1.0f;                                                                   \
        ZB = g_ * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;                        \
        sfpi::vFloat r_ = sfpi::approx_recip(p_);                                         \
        FB = sfpi::vConstFloatPrgm2 - ZB;                                                 \
        p_ = 1.0f - p_ * r_;                                                              \
        FB = g_ * sfpi::vConstFloatPrgm1 + FB;                                            \
        r_ = WA * r_;                                                                     \
        r_ = r_ * p_ + r_;                                                                \
        if constexpr (D16) {                                                              \
            r_ = sfpi::convert<sfpi::vFloat16b>(r_, sfpi::RoundMode::Nearest);            \
        }                                                                                 \
        sfpi::dst_reg[OOFF + (I)] = r_;                                                   \
    }

template <bool D16, int UOFF, int OOFF, bool HOIST_K3, bool HOIST_K2 = false, int ITERATIONS = 8>
inline void calculate_swiglu_spi_u() {
    static_assert(ITERATIONS == 8);
    HoistedIf<HOIST_K3> K3V = SWG_H_K3;  // an LReg (one SFPLOADI per row fewer) or a literal
    HoistedIf<HOIST_K2> K2V = SWG_H_K2;
    sfpi::vFloat wa, za, fa, wb, zb, fb;
    SWG_HEAD(0, wa, za, fa);
    SWG_STEP(0, wa, za, fa, wb, zb, fb);
    SWG_STEP(1, wb, zb, fb, wa, za, fa);
    SWG_STEP(2, wa, za, fa, wb, zb, fb);
    SWG_STEP(3, wb, zb, fb, wa, za, fa);
    SWG_STEP(4, wa, za, fa, wb, zb, fb);
    SWG_STEP(5, wb, zb, fb, wa, za, fa);
    SWG_STEP(6, wa, za, fa, wb, zb, fb);
    swg_tail<D16, UOFF, OOFF, 7>(wb, zb, fb);
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 2;
}

// spi2: spi with the degree-2 polynomial (one SFPMAD and one SFPLOADI fewer per row).
#define SWG_STEP2(I, WA, ZA, FA, WB, ZB, FB)                                              \
    {                                                                                     \
        sfpi::vFloat p_ = Q2V * FA + sfpi::vConstFloatPrgm0;                              \
        sfpi::vFloat g_ = sfpi::dst_reg[(I) + 1];                                         \
        p_ = p_ * FA + 1.0f;                                                              \
        WB = sfpi::dst_reg[UOFF + (I) + 1];                                               \
        ZA = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(ZA) << 23);                      \
        WB = WB * g_;                                                                     \
        p_ = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p_) + sfpi::as<sfpi::vInt>(ZA)); \
        g_ = sfpi::max(g_, -SWG_ARG_MAX);                                                 \
        p_ = p_ + 1.0f;                                                                   \
        g_ = sfpi::min(g_, SWG_ARG_MAX);                                                  \
        sfpi::vFloat r_ = sfpi::approx_recip(p_);                                         \
        ZB = g_ * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;                        \
        p_ = 1.0f - p_ * r_;                                                              \
        FB = sfpi::vConstFloatPrgm2 - ZB;                                                 \
        r_ = WA * r_;                                                                     \
        FB = g_ * sfpi::vConstFloatPrgm1 + FB;                                            \
        r_ = r_ * p_ + r_;                                                                \
        if constexpr (D16) {                                                              \
            r_ = sfpi::convert<sfpi::vFloat16b>(r_, sfpi::RoundMode::Nearest);            \
        }                                                                                 \
        sfpi::dst_reg[OOFF + (I)] = r_;                                                   \
    }

template <bool D16, int UOFF, int OOFF, bool HOIST_K2 = false, int ITERATIONS = 8>
inline void calculate_swiglu_spi2_u() {
    static_assert(ITERATIONS == 8);
    HoistedIf<HOIST_K2> Q2V = SWG_Q_K2;
    sfpi::vFloat wa, za, fa, wb, zb, fb;
    SWG_HEAD(0, wa, za, fa);
    SWG_STEP2(0, wa, za, fa, wb, zb, fb);
    SWG_STEP2(1, wb, zb, fb, wa, za, fa);
    SWG_STEP2(2, wa, za, fa, wb, zb, fb);
    SWG_STEP2(3, wb, zb, fb, wa, za, fa);
    SWG_STEP2(4, wa, za, fa, wb, zb, fb);
    SWG_STEP2(5, wb, zb, fb, wa, za, fa);
    SWG_STEP2(6, wa, za, fa, wb, zb, fb);
    swg_tail<D16, UOFF, OOFF, 7, 2>(wb, zb, fb);
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 2;
}

inline void swiglu_spi2_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = SWG_Q_K1;
    sfpi::vConstFloatPrgm1 = SWG_NEG_LOG2E;
    sfpi::vConstFloatPrgm2 = SWG_MAGIC;
}

// spi4: fp32-output grade. Degree-4 2^f (max rel err 2^-18.3; k3 / k4 fp16-exact, k1 in Prgm0) and two
// Newton steps on the 7-bit SFPARECIP seed (7 -> 14 -> 28 bits), the second folded into the output product.
constexpr float SWG_F_K1 = 0.6931241750717163f, SWG_F_K2 = 0.2402394413948059f, SWG_F_K3 = 0.055908203125f,
                SWG_F_K4 = 0.00959014892578125f;

#define SWG_STEP4(I, WA, ZA, FA, WB, ZB, FB)                                              \
    {                                                                                     \
        sfpi::vFloat p_ = SWG_F_K4 * FA + SWG_F_K3;                                       \
        sfpi::vFloat g_ = sfpi::dst_reg[(I) + 1];                                         \
        p_ = p_ * FA + SWG_F_K2;                                                          \
        WB = sfpi::dst_reg[UOFF + (I) + 1];                                               \
        p_ = p_ * FA + sfpi::vConstFloatPrgm0;                                            \
        WB = WB * g_;                                                                     \
        p_ = p_ * FA + 1.0f;                                                              \
        ZA = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(ZA) << 23);                      \
        g_ = sfpi::max(g_, -SWG_ARG_MAX);                                                 \
        p_ = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p_) + sfpi::as<sfpi::vInt>(ZA)); \
        g_ = sfpi::min(g_, SWG_ARG_MAX);                                                  \
        p_ = p_ + 1.0f;                                                                   \
        ZB = g_ * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;                        \
        sfpi::vFloat r_ = sfpi::approx_recip(p_);                                         \
        FB = sfpi::vConstFloatPrgm2 - ZB;                                                 \
        sfpi::vFloat e_ = 1.0f - p_ * r_;                                                 \
        FB = g_ * sfpi::vConstFloatPrgm1 + FB;                                            \
        r_ = r_ * e_ + r_;                                                                \
        e_ = 1.0f - p_ * r_;                                                              \
        r_ = WA * r_;                                                                     \
        r_ = r_ * e_ + r_;                                                                \
        if constexpr (D16) {                                                              \
            r_ = sfpi::convert<sfpi::vFloat16b>(r_, sfpi::RoundMode::Nearest);            \
        }                                                                                 \
        sfpi::dst_reg[OOFF + (I)] = r_;                                                   \
    }

template <bool D16, int OOFF, int I>
sfpi_inline void swg_tail4(sfpi::vFloat w, sfpi::vFloat z, sfpi::vFloat f) {
    sfpi::vFloat p = ((SWG_F_K4 * f + SWG_F_K3) * f + SWG_F_K2) * f + sfpi::vConstFloatPrgm0;
    p = p * f + 1.0f;
    sfpi::vFloat den = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + (sfpi::as<sfpi::vInt>(z) << 23)) + 1.0f;
    sfpi::vFloat r = sfpi::approx_recip(den);
    sfpi::vFloat e = 1.0f - den * r;
    r = r * e + r;
    e = 1.0f - den * r;
    r = w * r;
    r = r * e + r;
    if constexpr (D16) {
        r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[OOFF + I] = r;
}

template <bool D16, int UOFF, int OOFF, int ITERATIONS = 8>
inline void calculate_swiglu_spi4_u() {
    static_assert(ITERATIONS == 8);
    sfpi::vFloat wa, za, fa, wb, zb, fb;
    SWG_HEAD(0, wa, za, fa);
    SWG_STEP4(0, wa, za, fa, wb, zb, fb);
    SWG_STEP4(1, wb, zb, fb, wa, za, fa);
    SWG_STEP4(2, wa, za, fa, wb, zb, fb);
    SWG_STEP4(3, wb, zb, fb, wa, za, fa);
    SWG_STEP4(4, wa, za, fa, wb, zb, fb);
    SWG_STEP4(5, wb, zb, fb, wa, za, fa);
    SWG_STEP4(6, wa, za, fa, wb, zb, fb);
    swg_tail4<D16, OOFF, 7>(wb, zb, fb);
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 2;
}

inline void swiglu_spi4_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = SWG_F_K1;
    sfpi::vConstFloatPrgm1 = SWG_NEG_LOG2E;
    sfpi::vConstFloatPrgm2 = SWG_MAGIC;
}

inline void swiglu_fast_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 2.0f;
    sfpi::vConstFloatPrgm1 = SWG_NEG_LOG2E;
    sfpi::vConstFloatPrgm2 = SWG_MAGIC;
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_partials = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
constexpr uint32_t swiglu_rows = get_compile_time_arg_val(2);
constexpr uint32_t swiglu_cols = get_compile_time_arg_val(3);
constexpr uint32_t swiglu_stride = get_compile_time_arg_val(4);
constexpr uint32_t swiglu_up_base = get_compile_time_arg_val(5);
constexpr uint32_t epilogue_block_tiles = get_compile_time_arg_val(6);
constexpr uint32_t variant = get_compile_time_arg_val(7);
constexpr bool silu_fp32_precision = get_compile_time_arg_val(8) != 0;
constexpr uint32_t block_tiles = get_compile_time_arg_val(9);

// The op's local Silu chain element (verbatim).
template <bool fp32_precision, ckl::Dst Slot = ckl::Dst::D0>
struct Silu : ckl::UnaryOp<Silu<fp32_precision, Slot>, Slot> {
    static ALWI void init() { silu_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        silu_tile<fp32_precision && DST_ACCUM_MODE>(ckl::to_u32(Slot) + slot_offset);
    }
};

// Fused out = silu(In0) * In1 -> Out, one SFPU pass. impl 0 = same math as Silu, 1 = fast3, 2 = fast2,
// 3 = fast3 without Newton.
template <uint32_t impl, ckl::Dst In0 = ckl::Dst::D0, ckl::Dst In1 = ckl::Dst::D1, ckl::Dst Out = ckl::Dst::D0>
struct SiluMul : ckl::BinaryOp<SiluMul<impl, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() {
        if constexpr (impl == 0 || impl == 4) {
            MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_same_init)));
        } else if constexpr (impl == 8) {
            MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_sym_init<3>)));
        } else if constexpr (impl == 9) {
            MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_sym_init<2>)));
        } else if constexpr ((impl >= 10 && impl <= 13) || impl == 15 || impl == 16) {
            MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_symh_init)));
        } else if constexpr (impl == 14 || impl == 17) {
            MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_spi2_init)));
        } else if constexpr (impl == 18) {
            MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_spi4_init)));
        } else {
            MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_fast_init)));
        }
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        if constexpr (impl >= 4) {
            constexpr int UOFF = (int(ckl::to_u32(In1)) - int(ckl::to_u32(In0))) * 32;
            constexpr int OOFF = (int(ckl::to_u32(Out)) - int(ckl::to_u32(In0))) * 32;
            constexpr bool d16 = !DST_ACCUM_MODE;
            const uint32_t g = ckl::to_u32(In0) + slot_offset;
            if constexpr (impl == 4) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_same_u < silu_fp32_precision && DST_ACCUM_MODE,
                    d16,
                    UOFF,
                    OOFF,
                    8 >
                    , g, VectorMode::RC)));
            } else if constexpr (impl == 5) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_fast_u<3, 1, d16, UOFF, OOFF, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 6) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_fast_u<2, 1, d16, UOFF, OOFF, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 7) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_fast_u<3, 0, d16, UOFF, OOFF, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 8) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_sym_u<3, d16, UOFF, OOFF, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 9) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_sym_u<2, d16, UOFF, OOFF, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 10) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_symh_u<d16, UOFF, OOFF, 1, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 11) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_symh_u<d16, UOFF, OOFF, 2, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 12) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_sp_u<d16, UOFF, OOFF, 8>, g, VectorMode::RC)));
            } else if constexpr (impl == 13 || impl == 15 || impl == 16) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_spi_u < d16, UOFF, OOFF, impl >= 15, impl == 16, 8 >, g, VectorMode::RC)));
            } else if constexpr (impl == 18) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_spi4_u<d16, UOFF, OOFF, 8>, g, VectorMode::RC)));
            } else {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    sfpu::calculate_swiglu_spi2_u < d16, UOFF, OOFF, impl == 17, 8 >, g, VectorMode::RC)));
            }
            return;
        }
        const uint32_t a = ckl::to_u32(In0) + slot_offset, b = ckl::to_u32(In1) + slot_offset,
                       o = ckl::to_u32(Out) + slot_offset;
        if constexpr (impl == 0) {
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_swiglu_same,
                (silu_fp32_precision && DST_ACCUM_MODE, !DST_ACCUM_MODE, 8),
                a,
                b,
                o,
                VectorMode::RC)));
        } else if constexpr (impl == 1) {
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_swiglu_fast,
                (3, 1, !DST_ACCUM_MODE, 8),
                a,
                b,
                o,
                VectorMode::RC)));
        } else if constexpr (impl == 2) {
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_swiglu_fast,
                (2, 1, !DST_ACCUM_MODE, 8),
                a,
                b,
                o,
                VectorMode::RC)));
        } else {
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_swiglu_fast,
                (3, 0, !DST_ACCUM_MODE, 8),
                a,
                b,
                o,
                VectorMode::RC)));
        }
    }
};

#define PARTIALS_IN(...)                      \
    ckl::CopyTile<                            \
        ckl::input(                           \
            cb_partials,                      \
            ckl::WaitPolicy::None,            \
            ckl::PopPolicy::None,             \
            ckl::InputTileMapping::Block,     \
            ckl::DataFormatReconfig::Enabled, \
            ckl::TileAddressing::Strided),    \
        __VA_ARGS__>

ALWI void run_epilogue(DataflowBuffer& partials_dfb, DataflowBuffer& out_dfb) {
    using namespace ckl;
    const auto shape = IterationShape::grid(swiglu_rows, swiglu_cols).block_size(epilogue_block_tiles);
    if constexpr (variant == 0) {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, StridedTileRange{0, swiglu_stride}},
            Silu<silu_fp32_precision, Dst::D0>{},
            PARTIALS_IN(Dst::D1){partials_dfb, StridedTileRange{swiglu_up_base, swiglu_stride}},
            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
    } else if constexpr ((variant >= 1 && variant <= 4) || (variant >= 6 && variant <= 20)) {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, StridedTileRange{0, swiglu_stride}},
            PARTIALS_IN(Dst::D1){partials_dfb, StridedTileRange{swiglu_up_base, swiglu_stride}},
            SiluMul<(variant <= 4 ? variant - 1 : variant - 2), Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
    } else if constexpr (variant == 21) {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, StridedTileRange{0, swiglu_stride}},
            PARTIALS_IN(Dst::D1){partials_dfb, StridedTileRange{swiglu_up_base, swiglu_stride}},
            SwiGluMul < silu_fp32_precision && DST_ACCUM_MODE,
            Dst::D0,
            Dst::D1,
            Dst::D0 > {},
            PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
    } else {
        eltwise_chain(
            shape,
            PARTIALS_IN(Dst::D0){partials_dfb, StridedTileRange{0, swiglu_stride}},
            PARTIALS_IN(Dst::D1){partials_dfb, StridedTileRange{swiglu_up_base, swiglu_stride}},
            PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
    }
}

void kernel_main() {
    const uint32_t reps = get_arg_val<uint32_t>(0);
    // The op's startup is the matmul's (SrcOrder::Reverse, act/weight/out); the epilogue chain reconfigures
    // from there. Here: partials -> out directly.
    compute_kernel_hw_startup(cb_partials, cb_partials, cb_output_tiles);
    CircularBuffer partials_buf(cb_partials);
    DataflowBuffer partials_dfb(cb_partials), out_dfb(cb_output_tiles);

    // Resident sharded block: mark it available once (the op's matmul pushes it).
    partials_buf.reserve_back(block_tiles);
    partials_buf.push_back(block_tiles);
    partials_buf.wait_front(block_tiles);
    {
        MaybeDeviceZoneScope("swg_epilogue");
        for (uint32_t r = 0; r < reps; ++r) {
            run_epilogue(partials_dfb, out_dfb);
        }
    }
    partials_buf.pop_front(block_tiles);
}
