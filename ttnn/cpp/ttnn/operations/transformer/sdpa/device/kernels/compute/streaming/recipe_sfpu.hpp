// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fp32_state_sfpu.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// Compute the small online-softmax correction accurately with the full scale.
// The sum update subsequently forms correction - 1 for the L1 numerator update.
template <uint32_t scale_fp32>
inline void calculate_sdpa_exp_correction() {
#ifndef SDPA_RECIPE_FP32
    // Compensation replays leave automatic DST increment enabled. This
    // routine advances dst_reg explicitly; reset before using the SFPI loop.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
#endif
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
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(a));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, 13);
    TTI_SFPLOADI(0, 0x8, 0);
    TTI_SFPCONFIG(0, 14, 0);
}

// Two-score FP32 subtraction/grid/refinement. Keep subtraction separate
// from the grid MAD to preserve the selected FP32 rounding point.
inline void calculate_sdpa_negate_max() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat maximum = sfpi::dst_reg[0];
        sfpi::vFloat negative = -maximum;
        sfpi::dst_reg[0] = negative;
        sfpi::dst_reg++;
    }
}

template <uint32_t scale_fp32, bool init_only = false, bool reuse = false>
inline void calculate_sdpa_fused_sub_exp() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
#ifdef SDPA_RECIPE_FP32
#ifdef SDPA_RECIPE_ACCURATE
    constexpr uint32_t replay_length = 22;
    if constexpr (!init_only) {
        // L11 is programmable but SFPI reserves it for -1. This raw-instruction
        // body temporarily borrows it and restores -1 before returning.
        constexpr float grid_bias = 31.0f * 1024.0f - 4.0f * (32512.0f - 32500.818359375f);
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(grid_bias));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(grid_bias));
        TTI_SFPCONFIG(0, 11, 0);
    }
#else
    constexpr uint32_t replay_length = 27;
#endif
#else
    constexpr uint32_t replay_length = 27;
#endif
    if constexpr (!reuse) {
        constexpr float grid_a = 1024.0f * 1.4426950408889634f * __builtin_bit_cast(float, scale_fp32);
        constexpr float grid_b = 31.0f * 1024.0f - 4.0f * (32512.0f - 32500.818359375f);
        constexpr float a = -8.01081703839006e27f;
        constexpr float b = 5.39213985142374e28f;
        constexpr float c = -1.05653430577460e29f;
        constexpr float d = 1.38859037541378e29f;
        TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
        TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
        TTI_SFPLOADI(7, 0xA, sdpa_lo16(grid_a));
        TTI_SFPLOADI(7, 0x8, sdpa_hi16(grid_a));
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
        TTI_SFPCONFIG(0, 12, 0);
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
        TTI_SFPCONFIG(0, 13, 0);
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(d));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(d));
        TTI_SFPCONFIG(0, 14, 0);
        // Each replay handles corresponding vectors from both score tiles and
        // their common maximum. The last store alone advances the DST address.
        TTI_REPLAY(0, replay_length, init_only ? 0 : 1, 1);
#ifdef SDPA_RECIPE_FP32
#ifndef SDPA_RECIPE_ACCURATE
        TTI_SFPLOAD(2, 0, ADDR_MOD_6, 128);
#endif
#else
        TTI_SFPLOAD(2, 0, ADDR_MOD_6, 128);
#endif
        TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
        TTI_SFPLOAD(1, 0, ADDR_MOD_6, 64);
#ifdef SDPA_RECIPE_FP32
#ifdef SDPA_RECIPE_ACCURATE
        TTI_SFPMAD(7, 0, 11, 0, 0);
        TTI_SFPMAD(7, 1, 11, 1, 0);
#else
        TTI_SFPADD(10, 0, 2, 0, 2);
        TTI_SFPADD(10, 1, 2, 1, 2);
        TTI_SFPLOADI(2, 0xA, sdpa_lo16(grid_b));
        TTI_SFPLOADI(2, 0x8, sdpa_hi16(grid_b));
        TTI_SFPMAD(7, 0, 2, 0, 0);
        TTI_SFPMAD(7, 1, 2, 1, 0);
#endif
#else
        TTI_SFPADD(10, 0, 2, 0, 2);
        TTI_SFPADD(10, 1, 2, 1, 2);
        TTI_SFPLOADI(2, 0xA, sdpa_lo16(grid_b));
        TTI_SFPLOADI(2, 0x8, sdpa_hi16(grid_b));
        TTI_SFPMAD(7, 0, 2, 0, 0);
        TTI_SFPMAD(7, 1, 2, 1, 0);
#endif
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 7);
        TTI_SFP_STOCH_RND(0, 0, 1, 1, 1, 7);
        TTI_SFPSHFT(13, 0, 4, 5);
        TTI_SFPSHFT(13, 1, 5, 5);
        TTI_SFPSETSGN(0, 4, 0, 0);
        TTI_SFPSETSGN(1, 5, 1, 0);
        TTI_SFPSETEXP(127, 0, 2, 1);
        TTI_SFPSETEXP(127, 1, 3, 1);
        TTI_SFPMAD(2, 6, 12, 4, 0);
        TTI_SFPMAD(3, 6, 12, 5, 0);
        TTI_SFPMAD(2, 4, 13, 4, 0);
        TTI_SFPMAD(3, 5, 13, 5, 0);
        TTI_SFPMAD(2, 4, 14, 4, 0);
        TTI_SFPMAD(3, 5, 14, 5, 0);
        TTI_SFPMUL(0, 4, p_sfpu::LCONST_0, 0, 0);
        TTI_SFPMUL(1, 5, p_sfpu::LCONST_0, 1, 0);
        TTI_SFPSTORE(0, 0, ADDR_MOD_6, 0);
        TTI_SFPSTORE(1, 0, ADDR_MOD_7, 64);
    }
    if constexpr (!init_only) {
#pragma GCC unroll 8
        for (int i = reuse ? 0 : 1; i < 32; ++i) {
            lltt::replay(0, replay_length);
        }
#ifdef SDPA_RECIPE_FP32
#ifdef SDPA_RECIPE_ACCURATE
        TTI_SFPLOADI(0, 0xA, sdpa_lo16(-1.0f));
        TTI_SFPLOADI(0, 0x8, sdpa_hi16(-1.0f));
        TTI_SFPCONFIG(0, 11, 0);
#endif
#endif
    }
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

template <int iterations>
inline void calculate_sdpa_exp_stream_effective() {
    static_assert(iterations % 2 == 0);
    // Replay slots 0..7 retain the fast-exp macro's repeating four-vector pattern.
    // Slots 8..21 hold the two-chain refiner. Only the second store advances DST.
    // Fit (0.995 * 2^(m-1) + 1/128) / m on [1,2], with 2^96 folded
    // into the coefficients. The linear grid carries the exponent. PV and the
    // LoFi denominator consume the same six-bit effective SrcB weights.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);
#ifdef SDPA_RECIPE_FP32
#ifdef SDPA_RECIPE_ACCURATE
    // Relative fit to exp2(m-1), without HiFi2's six-bit rounding bias.
    constexpr float a = -8.01081703839006e27f;
    constexpr float b = 5.39213985142374e28f;
    constexpr float c = -1.05653430577460e29f;
    constexpr float d = 1.38859037541378e29f;
#else
    constexpr float a = -8.106702647031e27f;
    constexpr float b = 5.446591617222e28f;
    constexpr float c = -1.069245718291e29f;
    constexpr float d = 1.399030410717e29f;
#endif
#else
    constexpr float a = -8.106702647031e27f;
    constexpr float b = 5.446591617222e28f;
    constexpr float c = -1.069245718291e29f;
    constexpr float d = 1.399030410717e29f;
#endif
    TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(d));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(d));
    TTI_SFPCONFIG(0, 14, 0);
    TTI_REPLAY(8, 14, 1, 1);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(1, 0, ADDR_MOD_6, 2);
    TTI_SFPSETEXP(127, 0, 2, 1);
    TTI_SFPSETEXP(127, 1, 3, 1);
    TTI_SFPMAD(2, 6, 12, 4, 0);
    TTI_SFPMAD(3, 6, 12, 5, 0);
    TTI_SFPMAD(2, 4, 13, 4, 0);
    TTI_SFPMAD(3, 5, 13, 5, 0);
    TTI_SFPMAD(2, 4, 14, 4, 0);
    TTI_SFPMAD(3, 5, 14, 5, 0);
    TTI_SFPMUL(0, 4, p_sfpu::LCONST_0, 0, 0);
    TTI_SFPMUL(1, 5, p_sfpu::LCONST_0, 1, 0);
    TTI_SFPSTORE(0, 0, ADDR_MOD_6, 0);
    TTI_SFPSTORE(1, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, 14);
    }
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

template <int iterations>
inline void calculate_sdpa_exp_refine_loadmacro() {
    static_assert(iterations % 2 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);
    constexpr float a = -8.01081703839006e27f;
    constexpr float b = 5.39213985142374e28f;
    constexpr float c = -1.05653430577460e29f;
    constexpr float d = 1.38859037541378e29f;
    TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(d));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(d));
    TTI_SFPCONFIG(0, 14, 0);
    TTI_SFPMUL(4, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(5, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_REPLAY(8, 10, 1, 1);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_6, 2);
    TTI_SFPMAD(2, 6, 12, 4, 0);
    TTI_SFPMAD(3, 6, 12, 5, 0);
    TTI_SFPMAD(2, 4, 13, 4, 0);
    TTI_SFPMAD(3, 5, 13, 5, 0);
    TTI_SFPMAD(2, 4, 14, 4, 0);
    TTI_SFPMAD(3, 5, 14, 5, 0);
    TTI_SFPLOADMACRO(8, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(13, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, 10);
    }
    // UnitDelayKind is elapsed SFPU instructions, as configured by exp init.
    // Retire the final two scheduled multiplies/stores before reconfiguration.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Approximate exp and add a half-ULP bias for HiFi2's six-bit SrcB grid.
// The caller MUST form the denominator from these same effective FPU weights.

// Each state is hi + lo, with both parts stored as BF16. Preserve the FP32
// update and both nearest BF16 conversions:
// sum = (hi + lo) * correction + chunk; hi = bf16(sum); lo = bf16(sum - hi).
//
// PACK calls this once before a group of two-state updates. It owns macro
// slots 0..3, replay entries 0..14, and address modifiers 6/7 until that group
// finishes. The one-state denominator update may overwrite replay afterwards;
// the next group reinitializes it, and the next exp init restores exp's macros.
inline void init_sdpa_compensated_state_macros() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    // Templates 0/1 round fixed VC=L0/L3. Override VB with the load's
    // register, leaving VC intact; store that rounded register two instructions
    // later at the load's captured DST address.
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 12, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(0, 0, 3, 3, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, 0xA, 0x0000);
    TTI_SFPLOADI(0, 0x8, 0x1384);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, 0x8, 0x1385);
    TTI_SFPCONFIG(0, 5, 0);
    // Templates 2/3 add the loaded low component to the fixed high component.
    TTI_SFPADD(10, 0, 1, 14, 0);
    TTI_SFPADD(10, 3, 2, 15, 0);
    TTI_SFPCONFIG(0x600, 6, 1);
    TTI_SFPCONFIG(0x700, 7, 1);
    TTI_SFPCONFIG(0xF00, 8, 1);  // BF16 stores; delays count SFPU instructions.

    // Record only: no DST access occurs until a subsequent tile_regs_wait.
    // DST holds (hi0, lo0, chunk0, hi1, lo1, chunk1, correction).
    // L0/L3 retain full sums; L1/L2 hold rounded results; L4/L5 are chunks.
    // Keep macro-load destinations below L4: VDHi also encodes address bit 0.
    TTI_REPLAY(0, 15, 0, 1);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 384);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 192);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 128);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPMAD(1, 6, 4, 0, 0);
    TTI_SFPMAD(2, 6, 5, 3, 0);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 192);
    TTI_SFPADD(10, 0, 1, 0, 2);
    TTI_SFPADD(10, 3, 2, 3, 2);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 256);
}

// Same update as the paired numerator, but each denominator row owns its
// correction. Eight BF16 tiles fill one half-DST. Keep this 16-instruction
// program beside the numerator's 15 entries without reconfiguring macros.
inline void init_sdpa_compensated_sum_replay() {
    TTI_REPLAY(15, 16, 0, 1);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 384);
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 448);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 192);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 128);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPMAD(1, 6, 4, 0, 0);
    TTI_SFPMAD(2, 7, 5, 3, 0);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 192);
    TTI_SFPADD(10, 0, 1, 0, 2);
    TTI_SFPADD(10, 3, 2, 3, 2);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 256);
}

template <int pairs, bool separate_corrections = false>
inline void calculate_sdpa_compensated_state() {
    static_assert(pairs == 1 || pairs == 2);
    static_assert(!separate_corrections || pairs == 2);
    if constexpr (pairs == 2) {
#pragma GCC unroll 8
        for (int i = 0; i < 32; ++i) {
            TTI_REPLAY(separate_corrections ? 15 : 0, separate_corrections ? 16 : 15, 0, 0);
        }
        // Retire the last scheduled round/stores before PACK reads DST or
        // another SFPU operation overwrites the working registers.
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else {
        // Keep the original one-state schedule. The paired macro relies on
        // the other chain to separate dependent arithmetic and pending stores.
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
        TTI_REPLAY(0, 11, 1, 1);
        TTI_SFPLOAD(6, 0, ADDR_MOD_6, 192);
        TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
        TTI_SFPLOAD(1, 0, ADDR_MOD_6, 64);
        TTI_SFPLOAD(2, 0, ADDR_MOD_6, 128);
        TTI_SFPADD(10, 0, 1, 0, 0);
        TTI_SFPMAD(0, 6, 2, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPADD(10, 0, 1, 0, 2);
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(1, 0, ADDR_MOD_6, 0);
        TTI_SFPSTORE(0, 0, ADDR_MOD_7, 64);
#pragma GCC unroll 8
        for (int i = 1; i < 32; ++i) {
            TTI_REPLAY(0, 11, 0, 0);
        }
    }
}

inline void calculate_sdpa_zero_sum() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif
