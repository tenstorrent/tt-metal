// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Op-local copies of the three fused binary SFPU activations (SwiGLU-OAI, SiTU-GLU, clamped
// SiLU-GLU) for the bf16-dest path, rewritten for SFPU instruction count only.
//
// The activation phase is serial MATH-thread SFPU time (~1.4-3k cycles per output tile, 6-22% of
// a core's busy time on the binary-activation shapes), and the stock bodies spend a third of
// their instructions re-loading fp32 constants (two SFPLOADIs each) on every row. Here:
//   * constants that every row needs are loaded once per face call into LRegs, or once per
//     kernel into the free programmable CRegs (vConstFloatPrgm1/2; Prgm0 stays 2.0 for the
//     reciprocal);
//   * the v_if clamps of SwiGLU-OAI become SFPSWAP min/max (sfpi::min / sfpi::clamp).
// Every multiply, add, polynomial step, reciprocal iteration and rounding is the stock one, in
// the stock order, on the same fp32 values, so the result is bit-identical to the moe_gpt /
// api/compute versions for every non-NaN input (a NaN gate/up is clamped to the limit instead
// of propagating through the clamp). The fp32-dest path keeps calling the stock functions.

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

#if defined(SWIGLU_OAI)
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#elif defined(SITU_GLU)
#include "ckernel_sfpu_situ_glu.h"
#elif defined(CLAMPED_SILU_GLU)
#include "ckernel_sfpu_clamped_silu_glu.h"
#endif

namespace ckernel::sfpu::re_act {

// _sfpu_exp_21f_bf16_<true> (no bf16 rounding of y) with caller-provided constants: ln2r is
// 1/ln2, c0..c2 the EXP_21F_BF16 polynomial. Each may be a float or a vFloat already holding it.
template <typename L, typename C0, typename C1, typename C2>
sfpi_inline sfpi::vFloat exp_21f(sfpi::vFloat val, L ln2r, C0 c0, C1 c1, C2 c2) {
    sfpi::vFloat xlog2 = val * ln2r + 127.f;
    xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);
    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = PolynomialEvaluator::eval(frac, c0, c1, c2);
    return sfpi::setexp(frac, exponential_part);
}

constexpr float kOneLn2 = 1.4426950216293334961f;

#if defined(SWIGLU_OAI)
// calculate_swiglu (moe_gpt/device/kernels/swiglu_sfpu.h), bf16 dest.
// Prgm0 = 2.0 (reciprocal), Prgm1 = 1/ln2, Prgm2 = EXP_21F_BF16_C2.
template <int ITERATIONS, class Config>
inline void swiglu_bf16(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float alpha = Config::alpha;
    constexpr float clamp_limit = Config::clamp_limit;
    constexpr uint dst_tile_size = 32;
    const sfpi::vFloat c0 = EXP_21F_BF16_C0;
    const sfpi::vFloat c1 = EXP_21F_BF16_C1;
    const sfpi::vFloat alpha_v = alpha;

    // Unrolled, the body fits the SFPU replay buffer: sfpi records it once per face and replays
    // it, so TRISC1 issues ~4 instructions per row instead of the whole body.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        gate = sfpi::min(gate, clamp_limit);
        up = sfpi::clamp(up, -clamp_limit, clamp_limit);
        up = up + 1.0f;

        sfpi::vFloat alpha_gate = gate * alpha_v;
        sfpi::vFloat exp_neg_x = exp_21f(-alpha_gate, sfpi::vConstFloatPrgm1, c0, c1, sfpi::vConstFloatPrgm2);
        sfpi::vFloat sig = sfpu_reciprocal_iter<1>(1.0f + exp_neg_x);

        sfpi::vFloat glu = gate * sig;
        sfpi::vFloat result = up * glu;
        result = sfpi::convert<sfpi::vFloat16b>(result, RoundMode::Nearest);

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = SwiGLUConfigGPTOSS>
inline void binary_act(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    if constexpr (is_fp32_dest_acc_en) {
        calculate_swiglu<true, ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    } else {
        swiglu_bf16<ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    }
}

inline void binary_act_init() {
    swiglu_init();
    sfpi::vConstFloatPrgm1 = kOneLn2;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2;
}
#endif

#if defined(SITU_GLU)
// calculate_situ_glu (ckernel_sfpu_situ_glu.h), bf16 dest. tanh_init owns Prgm0..2 (the top
// three tanh coefficients), so the hoisted constants live in LRegs.
template <int ITERATIONS, class Config>
inline void situ_glu_bf16(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float beta_gate = Config::beta_gate;
    constexpr float inv_beta_gate = 1.0f / Config::beta_gate;
    constexpr float beta_up = Config::beta_up;
    constexpr float inv_beta_up = 1.0f / Config::beta_up;
    constexpr uint dst_tile_size = 32;
    const sfpi::vFloat t3 = TANH_POLY_C3;
    const sfpi::vFloat t2 = TANH_POLY_C2;
    const sfpi::vFloat t1 = TANH_POLY_C1;

    // _sfpu_tanh_polynomial_ with the two hoisted coefficients.
    auto tanh_poly = [&](sfpi::vFloat x) {
        sfpi::vFloat val = sfpi::abs(x);
        sfpi::vFloat result = PolynomialEvaluator::eval(
            val, 0.0f, t1, t2, t3, sfpi::vConstFloatPrgm2, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm0);
        result = sfpi::min(result, 1.0f);
        return sfpi::copysgn(result, x);
    };

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];

        sfpi::vFloat softcap_gate = tanh_poly(gate * inv_beta_gate) * beta_gate;
        sfpi::vFloat exp_neg_x = exp_21f(-gate, kOneLn2, EXP_21F_BF16_C0, EXP_21F_BF16_C1, EXP_21F_BF16_C2);
        sfpi::vFloat situ_a = softcap_gate * _situ_glu_reciprocal_<1>(1.0f + exp_neg_x);
        // up is loaded only now, so it holds no LReg through the gate half.
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        sfpi::vFloat result = situ_a * (tanh_poly(up * inv_beta_up) * beta_up);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = SituGluConfigKimi>
inline void binary_act(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    if constexpr (is_fp32_dest_acc_en) {
        calculate_situ_glu<true, ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    } else {
        situ_glu_bf16<ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    }
}

inline void binary_act_init() { situ_glu_init(); }
#endif

#if defined(CLAMPED_SILU_GLU)
// calculate_clamped_silu_glu (ckernel_sfpu_clamped_silu_glu.h), bf16 dest.
// _sfpu_sigmoid_<false>: Prgm0 = 2.0 (reciprocal), Prgm1 = 1/ln2, Prgm2 = EXP_21F_BF16_C2.
template <int ITERATIONS, class Config>
inline void clamped_silu_glu_bf16(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float limit = Config::limit;
    constexpr uint dst_tile_size = 32;
    const sfpi::vFloat c0 = EXP_21F_BF16_C0;
    const sfpi::vFloat c1 = EXP_21F_BF16_C1;

    // Replayed like swiglu_bf16's body.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        gate = sfpi::min(gate, limit);
        up = sfpi::clamp(up, -limit, limit);

        sfpi::vFloat exp_neg_x = exp_21f(-gate, sfpi::vConstFloatPrgm1, c0, c1, sfpi::vConstFloatPrgm2);
        sfpi::vFloat sig = sfpu_reciprocal_iter<1>(1.0f + exp_neg_x);
        sfpi::vFloat result = gate * sig * up;
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = ClampedSiluGluConfigDsV4>
inline void binary_act(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    if constexpr (is_fp32_dest_acc_en) {
        calculate_clamped_silu_glu<true, ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    } else {
        clamped_silu_glu_bf16<ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    }
}

inline void binary_act_init() {
    clamped_silu_glu_init();
    sfpi::vConstFloatPrgm1 = kOneLn2;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2;
}
#endif

}  // namespace ckernel::sfpu::re_act
