// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Pack-thread tanh GELU for linear_gelu (Blackhole, fp32 DEST): raw-LLK body behind GeluTanhActivation<
// kGeluLoadMacro>, routed through explicit specializations of the kernel_lib activation helpers.
//
// Helper bypass (measured, Perf 1): ckernel::sfpu::calculate_gelu_tanh_fast (GeluTanhActivation<1>) compiles to one
// serial dependency chain per 32-lane row, 34 issued SFPU instructions plus ~10 SFPMAD auto-stall cycles (~43.6
// cycles/row, 1031 ns/tile). This body runs two DEST rows in lockstep so each SFPMAD's 2-cycle latency is covered by
// the partner row, keeps the constants in LRegs, splits log2 into integer and fraction on the round sub-unit
// (SFPSTOCHRND), and issues each row's reload / multiply / store tail as one SFPLOADMACRO: 602 ns/tile, whole op
// 8x512 136.3 -> 98.1 us. The activation helpers take a closed KernelActivation enum, so a custom SFPU body can only
// be plugged in by specializing them. Do not replace this with the helper's GELU_TANH path without re-measuring.
//
// Same function and precision class as calculate_gelu_tanh_fast: x / (1 + 2^(x (A + B x^2))) with log2 clamped to
// [0, 254], the same degree-4 2^f polynomial, and 1/d from SFPARECIP refined by r (1 + e + e^2 + e^3), e = 1 - d r
// (error e^4, unbiased at fp32 like the two Newton steps it replaces).
//
// SFPU state owned on the pack thread (nothing else reprograms it between tiles; the math thread drives only the
// FPU): LReg11 = 254 (replaces vConstNeg1), LReg12-14 constants, load-macro InstructionTemplate[0..1], Sequence[0..1]
// and Misc, ADDR_MOD_6 = dest += 4. Requires fp32 DEST: there is no bf16 round before the store.
#pragma once

#include "vendor/kernel_lib/sfpu_activation_helpers.hpp"

#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_gelu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu::linear_gelu_lm {

constexpr std::uint32_t L_ZERO = p_sfpu::LCONST_0;
constexpr std::uint32_t L_ONE = p_sfpu::LCONST_1;
constexpr std::uint32_t L_254 = 11;  // programmable constant, set by gelu_init
constexpr std::uint32_t L_A = 12;    // vConstFloatPrgm0 = GELU_TANH_FAST_A
constexpr std::uint32_t L_B = 13;    // vConstFloatPrgm1 = GELU_TANH_FAST_B
constexpr std::uint32_t L_C1 = 14;   // vConstFloatPrgm2 = C1_U

constexpr std::uint16_t BF16_127 = 0x42FE;
constexpr std::uint16_t BF16_254 = 0x437E;
// GELU_TANH_FAST_EXP2_C{2,3,4} rescaled by 2^(23k): the fraction comes exact from the round sub-unit, not as a
// 23-bit integer mantissa. Exact power-of-two rescales, still bf16-exact.
constexpr std::uint16_t BF16_C2_U = 0x3E77;
constexpr std::uint16_t BF16_C3_U = 0x3D56;
constexpr std::uint16_t BF16_C4_U = 0x3C5C;
constexpr float C1_U = GELU_TANH_FAST_EXP2_C1 * 8388608.0f;

constexpr std::uint32_t NEG_A = 1;  // SFPMAD_MOD1_NEGATE_VA
constexpr std::uint32_t STOCH_RND_ZERO = 2;
constexpr std::uint32_t STOCH_TO_UINT8 = 2;

// Row A at dest +0 (x = L0, u = L1, v = L2), row B at dest +2 (x = L4, u = L5, v = L6); L3 / L7 hold transient
// constants shared by both rows. v must be L2 / L6: the load-macro templates name them.
inline void pair_body() {
    constexpr std::uint32_t AX = 0, AU = 1, AV = 2, BX = 4, BU = 5, BV = 6;
    constexpr std::uint32_t K0 = 3, K1 = 7;
    TTI_SFPLOADI(K0, 0, BF16_127);
    TTI_SFPLOAD(AX, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(BX, 0, ADDR_MOD_7, 2);
    TTI_SFPMUL(AX, AX, L_ZERO, AU, 0);  // x2
    TTI_SFPMUL(BX, BX, L_ZERO, BU, 0);
    TTI_SFPMAD(AU, L_B, L_A, AU, 0);  // t = x2 B + A
    TTI_SFPMAD(BU, L_B, L_A, BU, 0);
    TTI_SFPMAD(AX, AU, K0, AU, 0);  // xlog2 = x t + 127
    TTI_SFPMAD(BX, BU, K0, BU, 0);
    TTI_SFPLOADI(K0, 0, BF16_C4_U);
    TTI_SFPLOADI(K1, 0, BF16_C3_U);
    TTI_SFPSWAP(0, AU, L_ZERO, 1);  // max(xlog2, 0); the min lands in the read-only LReg9
    TTI_SFPSWAP(0, BU, L_ZERO, 1);
    TTI_SFPSWAP(0, L_254, AU, 1);  // min(xlog2, 254)
    TTI_SFPSWAP(0, L_254, BU, 1);
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, AU, AV, STOCH_TO_UINT8);  // E = trunc(xlog2)
    TTI_SFP_STOCH_RND(STOCH_RND_ZERO, 0, 9, BU, BV, STOCH_TO_UINT8);
    TTI_SFPCAST(AV, AX, 0);
    TTI_SFPCAST(BV, BX, 0);
    TTI_SFPMAD(AX, L_ONE, AU, AU, NEG_A);  // f = xlog2 - E
    TTI_SFPMAD(BX, L_ONE, BU, BU, NEG_A);
    TTI_SFPMAD(AU, K0, K1, AX, 0);  // 2^f, Horner
    TTI_SFPMAD(BU, K0, K1, BX, 0);
    TTI_SFPLOADI(K0, 0, BF16_C2_U);
    TTI_SFPMAD(AU, AX, K0, AX, 0);
    TTI_SFPMAD(BU, BX, K0, BX, 0);
    TTI_SFPMAD(AU, AX, L_C1, AX, 0);
    TTI_SFPMAD(BU, BX, L_C1, BX, 0);
    TTI_SFPMAD(AU, AX, L_ONE, AX, 0);
    TTI_SFPMAD(BU, BX, L_ONE, BX, 0);
    TTI_SFPSETEXP(0, AX, AV, 0);  // 2^E * 2^f
    TTI_SFPSETEXP(0, BX, BV, 0);
    TTI_SFPMAD(AV, L_ONE, L_ONE, AV, 0);  // d = e + 1
    TTI_SFPMAD(BV, L_ONE, L_ONE, BV, 0);
    TTI_SFPARECIP(0, AV, AU, 0);  // r ~ 1/d
    TTI_SFPARECIP(0, BV, BU, 0);
    TTI_SFPMAD(AV, AU, L_ONE, AX, NEG_A);  // eps = 1 - d r
    TTI_SFPMAD(BV, BU, L_ONE, BX, NEG_A);
    TTI_SFPMAD(AX, AX, AX, AV, 0);  // E2 = eps + eps^2 (d is dead)
    TTI_SFPMAD(BX, BX, BX, BV, 0);
    TTI_SFPMAD(AV, AX, AX, AX, 0);  // E3 = E2 eps + eps
    TTI_SFPMAD(BV, BX, BX, BX, 0);
    TTI_SFPMAD(AU, AX, AU, AU, 0);  // r' = r + r E3
    TTI_SFPMAD(BU, BX, BU, BU, 0);
    // Reload x into L2 / L6; the macro schedules VD = r' * VD the next cycle and stores VD two cycles later. The next
    // issued instruction must not target the MAD sub-unit (the scheduled MAD wins): it is the next pair's SFPLOADI.
    TTI_SFPLOADMACRO((0 << 2) | (AV & 3), 0, ADDR_MOD_7, 0 | (AV >> 2));
    TTI_SFPLOADMACRO((1 << 2) | (BV & 3), 0, ADDR_MOD_6, 2 | (BV >> 2));  // ADDR_MOD_6: dest += 4 (the pair)
}

inline void init_tail_macros() {
    // InstructionTemplate[0] / [1]: SFPMUL VA = r' of row A (L1) / row B (L5), VC = 0; VB and VD come from the macro.
    TTI_SFPMUL(1, 0, L_ZERO, 12, 0);
    TTI_SFPMUL(5, 0, L_ZERO, 13, 0);
    TTI_SFPNOP;
    constexpr std::uint32_t mad0 = 0x80 | (0 << 3) | (4 + 0);  // VB <- VD, delay 0, template 0
    constexpr std::uint32_t mad1 = 0x80 | (0 << 3) | (4 + 1);
    constexpr std::uint32_t store = (2 << 3) | 3;  // SFPSTORE VD, delay 2
    TTI_SFPLOADI(0, 10, (mad0 << 8));              // SFPLOADI_MOD0_LOWER
    TTI_SFPLOADI(0, 8, (store << 8));              // SFPLOADI_MOD0_UPPER
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPLOADI(0, 10, (mad1 << 8));
    TTI_SFPLOADI(0, 8, (store << 8));
    TTI_SFPCONFIG(0, 4 + 1, 0);
    // Misc: StoreMod0 0, UsesLoadMod0ForStore for macros 0 / 1, delays in cycles.
    TTI_SFPCONFIG(0x030, 8, 1);
    TTI_SFPNOP;
}

inline void gelu_init() {
    gelu_tanh_fast_init();
    TTI_SFPLOADI(0, 0, BF16_254);
    TTI_SFPCONFIG(0, L_254, 0);
    TTI_SFPNOP;
    sfpi::vConstFloatPrgm2 = C1_U;
    init_tail_macros();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

inline void gelu_face() {
#pragma GCC unroll 4
    for (int pair = 0; pair < 4; ++pair) {
        pair_body();
    }
}

}  // namespace ckernel::sfpu::linear_gelu_lm
#endif

// GeluTanhActivation param0 selecting this body (the library uses 0 = accurate, 1 = fast).
constexpr uint32_t kGeluLoadMacro = 2;

namespace compute_kernel_lib {

template <>
struct ActivationInitHelper<KernelActivation::GELU_TANH, kGeluLoadMacro, 0> {
    FORCE_INLINE static void init() {
        PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(ckernel::sfpu::linear_gelu_lm::gelu_init));
    }
};

template <>
struct ActivationApplyHelper<KernelActivation::GELU_TANH, kGeluLoadMacro, 0, 0> {
    FORCE_INLINE static void apply(uint32_t tile_index) {
        PACK(
            _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::linear_gelu_lm::gelu_face, tile_index, VectorMode::RC));
    }
};

}  // namespace compute_kernel_lib
