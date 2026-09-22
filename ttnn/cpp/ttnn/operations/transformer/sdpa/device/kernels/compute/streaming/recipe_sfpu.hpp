// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fp32_state_sfpu.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// Compute the small online-softmax correction accurately with the full scale.
// Rescale the prior numerator and denominator when the online maximum changes.
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

// Negate the maximum for FP32 L1 subtraction. Keep subtraction separate
// from the exponent grid MAD to preserve the selected FP32 rounding point.
inline void calculate_sdpa_negate_max() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat maximum = sfpi::dst_reg[0];
        sfpi::vFloat negative = -maximum;
        sfpi::dst_reg[0] = negative;
        sfpi::dst_reg++;
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
    constexpr float a = accurate ? -8.01081703839006e27f : -8.106702647031e27f;
    constexpr float b = accurate ? 5.39213985142374e28f : 5.446591617222e28f;
    constexpr float c = accurate ? -1.05653430577460e29f : -1.069245718291e29f;
    constexpr float d = accurate ? 1.38859037541378e29f : 1.399030410717e29f;
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

}  // namespace ckernel::sfpu
#endif
