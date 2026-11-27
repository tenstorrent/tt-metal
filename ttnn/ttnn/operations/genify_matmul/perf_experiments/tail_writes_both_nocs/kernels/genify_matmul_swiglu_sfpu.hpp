// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused swiglu epilogue element for eltwise_chain: out = silu(gate) * up in ONE SFPU pass over two DEST
// slots (Perf 1 graduation from perf_experiments/swiglu_epilogue_sfpu: isolated focus epilogue 85.3 -> 61.8 us,
// fp32 out 116.9 -> 75.6 us; error <= the replaced silu_tile + MulBinary chain on every metric).
// Helper bypass (capability): kernel_lib has no silu(a)*b element, and its binary SFPU path takes runtime DEST
// indices, while this fully unrolled row-pair-interleaved body needs compile-time row offsets. The chain
// itself is NOT bypassed (it still owns init, DEST sync, CB sync and format reconfig). Blackhole only
// (SFPARECIP via sfpi::approx_recip).
//
//   eltwise_chain(shape,
//       CopyTile<gate ...,  Dst::D0>{...},
//       CopyTile<up ...,    Dst::D1>{...},
//       SwiGluMul<fp32_grade, Dst::D0, Dst::D1, Dst::D0>{},
//       PackTile<...>{...});
//
// Replaces [Silu (silu_tile) on D0, MulBinary(D0, D1)] -- the chain element boundary (init / exec through
// the BinaryOp CRTP extension point) is unchanged; only the SFPU body is new. Why it is faster (BH, measured):
//  1. one SFPU pass instead of two (no MulBinary load/load/mul/store pass, no intermediate bf16 round);
//  2. sigmoid(g) = 1 / (1 + E), E = 2^(-g log2e) with g clamped to [-87, 87]: E is a normal fp32 in
//     [2^-126, 2^126), so the 7-bit SFPARECIP seed never hits its 0 / inf cases and Newton needs no NaN
//     guard (no v_if), and there is no |g| / sign select. 2^t = magic-number round (t + 1.5*2^23),
//     a polynomial in the fraction, and an integer add of n << 23 into the exponent;
//  3. the Newton step is r + r*(1 - d*r), folded into the output product (out = q + q*eps, q = g*u*r);
//  4. rows are software-pipelined in pairs at STATEMENT level: the tail of row i (polynomial, exponent,
//     reciprocal, Newton, store) is interleaved with the head of row i+1 (loads, g*u, clamp, z, f), so
//     two independent SFPMAD dependency chains fill each other's latency shadows. sfpi keeps source order
//     (it does not schedule across rows itself; writing the same rows sequentially measured ~11% slower).
//  5. DEST addressing goes through the UNARY params wrapper (base = gate slot, up / out at compile-time row
//     offsets), so every SFPLOAD / SFPSTORE is a static word (measured neutral, but required for 4's
//     fully-unrolled static row offsets).
// Precision grades (template `fp32_grade`, the op's silu_fp32_precision knob):
//   false: degree-3 2^f (max rel err 2^-13.1), one Newton step -> for non-fp32 outputs.
//   true:  degree-4 2^f (2^-18.3), two Newton steps (7 -> 14 -> 28 bits) -> for fp32 outputs.
// Program constants: Prgm0 = k1, Prgm1 = -log2(e), Prgm2 = 1.5 * 2^23 (set by init; Prgm0's 2.0 for the
// stock reciprocal is NOT kept, so do not interleave stock sigmoid/recip SFPU code between init and exec).

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_sigmoid.h"  // math::reset_counters / p_setrwc via cmath_common
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

constexpr float SWGLU_NEG_LOG2E = -1.4426950408889634f;
constexpr float SWGLU_MAGIC = 12582912.0f;  // 1.5 * 2^23
constexpr float SWGLU_ARG_MAX = 87.0f;      // bf16-exact: one SFPLOADI
// bf16 grade, degree 3 (c0 = 1; k2, k3 fp16-exact -> one SFPLOADI each; k1 in Prgm0)
constexpr float SWGLU_B_K1 = 0.6932584643363953f, SWGLU_B_K2 = 0.2423095703125f, SWGLU_B_K3 = 0.055328369140625f;
// fp32 grade, degree 4 (c0 = 1; k3, k4 fp16-exact; k1 in Prgm0)
constexpr float SWGLU_F_K1 = 0.6931241750717163f, SWGLU_F_K2 = 0.2402394413948059f, SWGLU_F_K3 = 0.055908203125f,
                SWGLU_F_K4 = 0.00959014892578125f;

// head(row I): w = g * u, x = clamp(g), z = x * -log2e + M (n = round(t) in z's low bits), f = t - n.
#define SWGLU_HEAD(I, W, Z, F)                                                     \
    {                                                                              \
        sfpi::vFloat g_ = sfpi::dst_reg[I];                                        \
        W = g_ * sfpi::dst_reg[UOFF + (I)];                                        \
        sfpi::vFloat x_ = sfpi::min(sfpi::max(g_, -SWGLU_ARG_MAX), SWGLU_ARG_MAX); \
        Z = x_ * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;                  \
        F = x_ * sfpi::vConstFloatPrgm1 + (sfpi::vConstFloatPrgm2 - Z);            \
    }

// tail(row I) of (WA, ZA, FA) interleaved with head(row I + 1) into (WB, ZB, FB). bf16 grade.
#define SWGLU_STEP_B(I, WA, ZA, FA, WB, ZB, FB)                                           \
    {                                                                                     \
        sfpi::vFloat p_ = K3V * FA + SWGLU_B_K2;                                          \
        sfpi::vFloat g_ = sfpi::dst_reg[(I) + 1];                                         \
        p_ = p_ * FA + sfpi::vConstFloatPrgm0;                                            \
        WB = sfpi::dst_reg[UOFF + (I) + 1];                                               \
        p_ = p_ * FA + 1.0f;                                                              \
        WB = WB * g_;                                                                     \
        ZA = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(ZA) << 23);                      \
        g_ = sfpi::max(g_, -SWGLU_ARG_MAX);                                               \
        p_ = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p_) + sfpi::as<sfpi::vInt>(ZA)); \
        g_ = sfpi::min(g_, SWGLU_ARG_MAX);                                                \
        p_ = p_ + 1.0f;                                                                   \
        ZB = g_ * sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm2;                        \
        sfpi::vFloat r_ = sfpi::approx_recip(p_);                                         \
        FB = sfpi::vConstFloatPrgm2 - ZB;                                                 \
        p_ = 1.0f - p_ * r_;                                                              \
        FB = g_ * sfpi::vConstFloatPrgm1 + FB;                                            \
        r_ = WA * r_;                                                                     \
        r_ = r_ * p_ + r_;                                                                \
        if constexpr (dest16) {                                                           \
            r_ = sfpi::convert<sfpi::vFloat16b>(r_, sfpi::RoundMode::Nearest);            \
        }                                                                                 \
        sfpi::dst_reg[OOFF + (I)] = r_;                                                   \
    }

// fp32 grade: degree-4 polynomial, two Newton steps.
#define SWGLU_STEP_F(I, WA, ZA, FA, WB, ZB, FB)                                           \
    {                                                                                     \
        sfpi::vFloat p_ = SWGLU_F_K4 * FA + SWGLU_F_K3;                                   \
        sfpi::vFloat g_ = sfpi::dst_reg[(I) + 1];                                         \
        p_ = p_ * FA + SWGLU_F_K2;                                                        \
        WB = sfpi::dst_reg[UOFF + (I) + 1];                                               \
        p_ = p_ * FA + sfpi::vConstFloatPrgm0;                                            \
        WB = WB * g_;                                                                     \
        p_ = p_ * FA + 1.0f;                                                              \
        ZA = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(ZA) << 23);                      \
        g_ = sfpi::max(g_, -SWGLU_ARG_MAX);                                               \
        p_ = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p_) + sfpi::as<sfpi::vInt>(ZA)); \
        g_ = sfpi::min(g_, SWGLU_ARG_MAX);                                                \
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
        if constexpr (dest16) {                                                           \
            r_ = sfpi::convert<sfpi::vFloat16b>(r_, sfpi::RoundMode::Nearest);            \
        }                                                                                 \
        sfpi::dst_reg[OOFF + (I)] = r_;                                                   \
    }

// Last row's tail (no next row to interleave with).
template <bool fp32_grade, bool dest16, int OOFF, int I>
sfpi_inline void swiglu_glu_tail(sfpi::vFloat w, sfpi::vFloat z, sfpi::vFloat f) {
    sfpi::vFloat p;
    if constexpr (fp32_grade) {
        p = ((SWGLU_F_K4 * f + SWGLU_F_K3) * f + SWGLU_F_K2) * f + sfpi::vConstFloatPrgm0;
    } else {
        p = (SWGLU_B_K3 * f + SWGLU_B_K2) * f + sfpi::vConstFloatPrgm0;
    }
    p = p * f + 1.0f;
    sfpi::vFloat den = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(p) + (sfpi::as<sfpi::vInt>(z) << 23)) + 1.0f;
    sfpi::vFloat r = sfpi::approx_recip(den);
    sfpi::vFloat e = 1.0f - den * r;
    if constexpr (fp32_grade) {
        r = r * e + r;
        e = 1.0f - den * r;
    }
    r = w * r;
    r = r * e + r;
    if constexpr (dest16) {
        r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[OOFF + I] = r;
}

// One face (8 SFPU rows); the unary params wrapper walks the 4 faces (VectorMode::RC).
// UOFF / OOFF: up / out row offsets from the gate slot (slot distance * 32).
template <bool fp32_grade, bool dest16, int UOFF, int OOFF, int ITERATIONS = 8>
inline void calculate_swiglu_glu() {
    static_assert(ITERATIONS == 8, "one face per call");
    [[maybe_unused]] HoistedIf<!fp32_grade> K3V = SWGLU_B_K3;  // the bf16 grade keeps k3 in an LReg
    sfpi::vFloat wa, za, fa, wb, zb, fb;
    SWGLU_HEAD(0, wa, za, fa);
    if constexpr (fp32_grade) {
        SWGLU_STEP_F(0, wa, za, fa, wb, zb, fb);
        SWGLU_STEP_F(1, wb, zb, fb, wa, za, fa);
        SWGLU_STEP_F(2, wa, za, fa, wb, zb, fb);
        SWGLU_STEP_F(3, wb, zb, fb, wa, za, fa);
        SWGLU_STEP_F(4, wa, za, fa, wb, zb, fb);
        SWGLU_STEP_F(5, wb, zb, fb, wa, za, fa);
        SWGLU_STEP_F(6, wa, za, fa, wb, zb, fb);
    } else {
        SWGLU_STEP_B(0, wa, za, fa, wb, zb, fb);
        SWGLU_STEP_B(1, wb, zb, fb, wa, za, fa);
        SWGLU_STEP_B(2, wa, za, fa, wb, zb, fb);
        SWGLU_STEP_B(3, wb, zb, fb, wa, za, fa);
        SWGLU_STEP_B(4, wa, za, fa, wb, zb, fb);
        SWGLU_STEP_B(5, wb, zb, fb, wa, za, fa);
        SWGLU_STEP_B(6, wa, za, fa, wb, zb, fb);
    }
    swiglu_glu_tail<fp32_grade, dest16, OOFF, 7>(wb, zb, fb);
    sfpi::dst_reg += 3;  // += 8 rows, in increments the SFPINCRWC encoding accepts
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 2;
}

#undef SWGLU_HEAD
#undef SWGLU_STEP_B
#undef SWGLU_STEP_F

template <bool fp32_grade>
inline void swiglu_glu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = fp32_grade ? SWGLU_F_K1 : SWGLU_B_K1;
    sfpi::vConstFloatPrgm1 = SWGLU_NEG_LOG2E;
    sfpi::vConstFloatPrgm2 = SWGLU_MAGIC;
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

namespace compute_kernel_lib {

// out(Out) = silu(In0) * In1. In1 and Out are addressed as compile-time offsets from In0.
template <bool fp32_grade, Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SwiGluMul : BinaryOp<SwiGluMul<fp32_grade, In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::swiglu_glu_init<fp32_grade>))); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        constexpr int UOFF = (int(to_u32(In1)) - int(to_u32(In0))) * 32;
        constexpr int OOFF = (int(to_u32(Out)) - int(to_u32(In0))) * 32;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_swiglu_glu<fp32_grade, !DST_ACCUM_MODE, UOFF, OOFF, 8>,
            to_u32(In0) + slot_offset,
            VectorMode::RC)));
    }
};

}  // namespace compute_kernel_lib
