// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fp32_state_sfpu.hpp"
#include "recipe_config.hpp"
// SFPU exponentials for the FP32-state recipes (BALANCED, ACCURATE); STANDARD and FAST use the
// standard approximate exp.
//
// Correction: c = exp(scale * (m_old - m_new)) with the accurate FP32 exp (calculate_sdpa_exp_correction).
//
// Scores: P = exp(scale * (s - m)) with s - m <= 0, in two steps.
//   1. Grid (init_sdpa_exp_grid, calculate_sdpa_exp_grid_batch): Schraudolph's bit trick.
//        y = round_int16(a * x + b),  a = 1024 * log2(e) * scale,  b = 31699.2734375
//        g = bits_as_fp32(y << 13)
//      The integer y lands in the FP32 exponent and top ten fraction bits, so g = 2^(y/1024 - 127)
//      piecewise linearly: g = 2^(E - 127) * m with m in [1, 2). y < 0 (scale * x < about -21.46) packs
//      as zero through packer ReLU. The bit trick needs x <= 0; it is not a general exp.
//   2. Refinement (calculate_sdpa_exp_refine_loadmacro): the linear mantissa m is replaced by 2^(m - 1)
//      with a polynomial, P = g * p(m), p(m) ~= 2^96 * 2^(m - 1) / m: a cubic for ACCURATE, a minimax
//      quadratic for BALANCED.
//   Overall P ~= K * exp(scale * x) with a constant K ~= 0.970 (the offset in b and the 2^96 scaling do not
//   cancel exactly). K cancels in O / l because P feeds both; P, l or an LSE produced here must not be
//   combined with values from a different exp implementation. Emulating p over m in [1, 2), the relative
//   ripple around K is +-0.35% for BALANCED's quadratic and +-0.10% for ACCURATE's cubic.
// SFPI code exists in the MATH and PACK builds only.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
// Raw SFPU load/store address modes. Wormhole encodes two bits; its SFPU launcher's base offset selects 4-7.
constexpr uint32_t kSdpaSfpuAddrMod6 = recipe::wormhole ? ADDR_MOD_2 : ADDR_MOD_6;
constexpr uint32_t kSdpaSfpuAddrMod7 = recipe::wormhole ? ADDR_MOD_3 : ADDR_MOD_7;

// Attention sink (recipe::sink): dest tile 0's first column (a row's denominator l) += k * exp(scale * (sink - m)),
// with m in dest tile 1's first column (the maxima the row's P were taken against). k is the score exp's mean factor
// (on average P = k * exp(scale * (s - m)), and l sums P), so the sink weighs like one more key with logit `sink`.
template <uint32_t scale_fp32, uint32_t k_bits>
inline void calculate_sdpa_sink_denominator(uint32_t sink_bits) {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float scale = __builtin_bit_cast(float, scale_fp32);
    constexpr float k = __builtin_bit_cast(float, k_bits);
    const sfpi::vFloat sink = Converter::as_float(sink_bits);
#pragma GCC unroll 1
    for (int d = 0; d < 4; ++d) {
        sfpi::vFloat l = sfpi::dst_reg[0];
        sfpi::vFloat m = sfpi::dst_reg[32];
        sfpi::dst_reg[0] = l + k * _sfpu_exp_fp32_accurate_((sink - m) * scale);
        sfpi::dst_reg += 2;
    }
}

// Online-softmax correction, accurate FP32 exp with the full scale.
template <uint32_t scale_fp32>
inline void calculate_sdpa_exp_correction() {
    if constexpr (!recipe::fp32) {
        // The score exp's SFPU programs may leave a DST increment in ADDR_MOD_7. This
        // routine advances dst_reg explicitly; reset before using the SFPI loop.
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    }
    constexpr float scale = __builtin_bit_cast(float, scale_fp32);
    for (int d = 0; d < 4; ++d) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = _sfpu_exp_fp32_accurate_(x * scale);
        sfpi::dst_reg += 2;
    }
}

// Nonpositive-logit grid: trade exponent range for 10 fraction bits
// in the fast-exp macro's signed-magnitude INT16 encoding. Values below about
// -21.45 after scaling are zeroed by packer ReLU. This is not a general exp.
template <uint32_t scale_fp32>
inline void init_sdpa_exp_grid() {
    constexpr float a = 1024.0f * 1.4426950408889634f * __builtin_bit_cast(float, scale_fp32);
    constexpr float b = 31.0f * 1024.0f - 4.0f * (32512.0f - 32500.818359375f);
    TTI_SFPLOADI(0, 0xA, Converter::lo16(a));
    TTI_SFPLOADI(0, 0x8, Converter::hi16(a));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, Converter::lo16(b));
    TTI_SFPLOADI(0, 0x8, Converter::hi16(b));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, 13);
    TTI_SFPLOADI(0, 0x8, 0);
    TTI_SFPCONFIG(0, 14, 0);
}

// Negate a tile. ACCURATE negates the row maximum so the packer's FP32 L1 accumulate computes s - m exactly in
// FP32, separately from the grid MAD (which would otherwise fold the subtraction into a * x + b); the fused chunks
// build their -e0 operand from the zeroed column identity.
inline void calculate_sdpa_negate_max() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat maximum = sfpi::dst_reg[0];
        sfpi::vFloat negative = -maximum;
        sfpi::dst_reg[0] = negative;
        sfpi::dst_reg++;
    }
}

// STANDARD and FAST score exp, one SFPLOADMACRO per 32 scores (call after the LLK fast exp's init,
// exp_init<true, scale, false>, which sets LReg14 = 15 and the MAD template L12 * x + L13).
//
// The LLK fast exp computes i = round(A x + B') with A = 256 log2(e) scale, STOCHRNDs it to INT16 and stores the
// sign-magnitude pattern i << 15 (P = 2^(i/256 - 127)); B' = 32500.818359375 - 256 * headroom_octaves. That takes two
// round-unit instructions (STOCHRND, SHFT2) and two issued instructions per 32 scores, and below about -88 + headroom
// it emits negative P for the packer's ReLU to clear. Here the MAD computes y = (2^24 - i) 2^104 instead
// (L12 = -A 2^104, L13 = (2^24 - B') 2^104, exact scalings):
//   - 0 < i <= 2^23: y lies in the top binade [2^127, 2^128), whose ulp is 2^104, so the FMA's round to nearest even
//     rounds i to an integer and bits(y) = 0x7F800000 - i;
//   - i <= 0 (P below 2^-127, at any distance below the maximum, including masked -2^100 logits): y overflows to +Inf
//     (0x7F800000; a NaN score gives a NaN with zero low bits on Blackhole).
// SHFT2 << L14 keeps the low 17 bits, w = ((-i) mod 2^17) << 15 (0 for +Inf), and an integer absolute value (Blackhole
// SFPCAST mode 2, Wormhole SFPABS integer mode; both simple-unit) gives |w| = i << 15: the LLK pattern for
// 0 < i < 2^16, and 0 for every i <= 0. In range the values equal the LLK exp's (one rounding, the FMA's, instead of
// the FMA's then STOCHRND's: a tie can round the other way). No negative P (the packer ReLU never fires), no wrap
// below the maximum. Above it, i >= 2^15 (P >= 2, about 0.72 + headroom above the reference max) gives large P
// whose row sums trip the fused chunks' redo check up to i of about 98800 (about 199 nats above the reference max
// in units of the scaled scores); past that w wraps (period 2^17), so P there is unspecified (the LLK exp saturates
// at P = 2 instead). Scaled logits that far above a Q chunk's first K chunk's row maximum are outside STANDARD's and
// FAST's range.
//
// Sequence 0, cycle-timed (Misc = 0): load, MAD (delay 0, template 1), SHFT2 (delay 2 = MAD latency, template 2,
// -> VD), integer abs (delay 3, template 3, -> LReg16), store from LReg16 (delay 4). The LOADMACROs rotate LRegs
// 0-7 (VDHi = Imm10 bit 0), so a register is reloaded only after its store has read it. The 32 recorded LOADMACROs
// cover one tile of a 16-bit DEST (64 rows, DEST += 2 per LOADMACRO).
template <uint32_t scale_fp32, uint32_t headroom_octaves>
inline void init_sdpa_score_exp() {
    constexpr float a = -(256.0f * 1.4426950408889634f * __builtin_bit_cast(float, scale_fp32)) * 0x1p104f;
    constexpr float c = (16777216.0f - (32500.818359375f - 256.0f * static_cast<float>(headroom_octaves))) * 0x1p104f;
    TTI_SFPLOADI(0, 0xA, Converter::lo16(a));
    TTI_SFPLOADI(0, 0x8, Converter::hi16(a));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, Converter::lo16(c));
    TTI_SFPLOADI(0, 0x8, Converter::hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, 14, 14, 5);  // template 2 (VD 14 backdoor): VB << LReg14
    TTI_SFPNOP;
    if constexpr (recipe::wormhole) {
        TTI_SFPABS(0, 0, 15, 0);  // template 3 (VD 15 backdoor): two's complement absolute value
    } else {
        TTI_SFPCAST(0, 15, 2);  // template 3 (VD 15 backdoor): int32 absolute value
    }
    TTI_SFPNOP;
    // Simple 0x5F (VC = VD, -> LReg16, delay 3, template 3), MAD 0x85 (VB = loaded, delay 0, template 1), round 0x96
    // (VB = VD, -> VD, delay 2, template 2), store 0x63 (from LReg16, delay 4).
    TTI_SFPLOADI(0, 0xA, 0x855F);
    TTI_SFPLOADI(0, 0x8, 0x6396);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0, 8, 1);  // Misc = 0: store in DEST's format, cycle-timed delays
    TTI_SFPNOP;
    auto record = [] {
#pragma GCC unroll 4
        for (uint32_t r = 0; r < 4; ++r) {
            TTI_SFPLOADMACRO(0, 0, kSdpaSfpuAddrMod7, 0);
            TTI_SFPLOADMACRO(1, 0, kSdpaSfpuAddrMod7, 0);
            TTI_SFPLOADMACRO(2, 0, kSdpaSfpuAddrMod7, 0);
            TTI_SFPLOADMACRO(3, 0, kSdpaSfpuAddrMod7, 0);
            TTI_SFPLOADMACRO(0, 0, kSdpaSfpuAddrMod7, 1);
            TTI_SFPLOADMACRO(1, 0, kSdpaSfpuAddrMod7, 1);
            TTI_SFPLOADMACRO(2, 0, kSdpaSfpuAddrMod7, 1);
            TTI_SFPLOADMACRO(3, 0, kSdpaSfpuAddrMod7, 1);
        }
    };
    // Blackhole records through load_replay_buf (instruction gathering); Wormhole records directly.
#ifdef ARCH_BLACKHOLE
    load_replay_buf(0, 32, record);
#else
    lltt::record(0, 32);
    record();
#endif
}

// `tiles` consecutive 16-bit DEST tiles from the current SFPU base (init_sdpa_score_exp).
inline void calculate_sdpa_score_exp(uint32_t tiles) {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
#pragma GCC unroll 1
    for (uint32_t t = 0; t < tiles; ++t) {
        lltt::replay(0, 32);
    }
    // The last LOADMACRO's store executes five cycles after it.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <int iterations>
inline void calculate_sdpa_exp_grid_batch() {
    static_assert(iterations % 4 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
#pragma GCC unroll 8
    for (int i = 0; i < iterations / 4; ++i) {
        lltt::replay(0, 8);
    }
    TTI_SFPNOP;
    TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline void init_sdpa_refine_loadmacros() {
    // Slot 0 is unused by the unclamped exp grid. SETEXP reads VC, so its
    // sequence must override VC (not VB) with the loaded register.
    TTI_SFPSETEXP(127, 0, 12, 1);
    TTI_SFPLOADI(0, 0xA, 0x0004);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);
    // Final load/multiply/store: polynomial is in L4/L5. The macro uses the
    // loaded linear grid value as VB and captures its DST address for store.
    TTI_SFPLOADI(0, 0xA, 0x8500);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 6, 0);
    TTI_SFPLOADI(0, 0xA, 0x8600);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 7, 0);
}

inline void restore_sdpa_grid_macro_instructions() {
    TTI_SFPMAD(12, 0, 13, 13, 0);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 7);
}

template <int iterations, bool accurate>
inline void calculate_sdpa_exp_refine_loadmacro() {
    static_assert(iterations % 2 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);
    // p(m) by Horner: leading coefficient in L6, the rest in the constant registers L12.. (ACCURATE: cubic,
    // BALANCED: quadratic, two multiply-adds fewer per row pair).
    constexpr float a = accurate ? -8.01081703839006e27f : 1.852119366802e+28f;
    constexpr float b = accurate ? 5.39213985142374e28f : -5.501756027717e+28f;
    constexpr float c = accurate ? -1.05653430577460e29f : 1.154751922200e+29f;
    constexpr uint32_t replay_len = accurate ? 10 : 8;
    TTI_SFPLOADI(6, 0xA, Converter::lo16(a));
    TTI_SFPLOADI(6, 0x8, Converter::hi16(a));
    TTI_SFPLOADI(0, 0xA, Converter::lo16(b));
    TTI_SFPLOADI(0, 0x8, Converter::hi16(b));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, Converter::lo16(c));
    TTI_SFPLOADI(0, 0x8, Converter::hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    if constexpr (accurate) {
        constexpr float d = 1.38859037541378e29f;
        TTI_SFPLOADI(0, 0xA, Converter::lo16(d));
        TTI_SFPLOADI(0, 0x8, Converter::hi16(d));
        TTI_SFPCONFIG(0, 14, 0);
    }
    TTI_SFPMUL(4, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(5, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_REPLAY(8, replay_len, 1, 1);
    TTI_SFPLOADMACRO(6, 0, kSdpaSfpuAddrMod6, 0);
    TTI_SFPLOADMACRO(7, 0, kSdpaSfpuAddrMod6, 2);
    TTI_SFPMAD(2, 6, 12, 4, 0);
    TTI_SFPMAD(3, 6, 12, 5, 0);
    TTI_SFPMAD(2, 4, 13, 4, 0);
    TTI_SFPMAD(3, 5, 13, 5, 0);
    if constexpr (accurate) {
        TTI_SFPMAD(2, 4, 14, 4, 0);
        TTI_SFPMAD(3, 5, 14, 5, 0);
    }
    TTI_SFPLOADMACRO(8, 0, kSdpaSfpuAddrMod6, 0);
    TTI_SFPLOADMACRO(13, 0, kSdpaSfpuAddrMod7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, replay_len);
    }
    // UnitDelayKind is elapsed SFPU instructions, as configured by exp init.
    // Retire the final two scheduled multiplies/stores before reconfiguration.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

}  // namespace ckernel::sfpu
#endif
