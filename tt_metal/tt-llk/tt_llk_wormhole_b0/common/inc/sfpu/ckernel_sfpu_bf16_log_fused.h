// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16
{
constexpr std::uint32_t kLogFusedHold    = ADDR_MOD_3;
constexpr std::uint32_t kLogFusedAdvance = ADDR_MOD_2;

template <std::uint32_t Register, std::uint32_t Bits>
inline void log_fused_load_constant()
{
    TTI_SFPLOADI(Register, sfpi::SFPLOADI_MOD0_UPPER, Bits >> 16);
    TTI_SFPLOADI(Register, sfpi::SFPLOADI_MOD0_LOWER, Bits & 0xffffu);
}

template <typename Config, int Iterations = 8>
inline void calculate_log_fused()
{
    static_assert(Iterations > 0);
    static_assert(Config::kDegree == 4u || Config::kDegree == 5u);
    // The row forms -e, so the exponent term multiplies by -scale; base 2 takes the -1 constant.
    constexpr bool unit_scale = Config::kScaleBits == 0x3f800000u;
    static_assert(Config::kDegree == 4u || unit_scale, "degree 5 needs a free register for -scale");
    constexpr std::uint32_t D   = Config::kDegree;
    constexpr std::uint32_t c[] = {p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG7};
    // c[0] holds the leading coefficient; c[D - j] holds coefficient j for j = 1 .. D - 1.
    log_fused_load_constant<c[0], Config::kCoefficientBits[D]>();
    log_fused_load_constant<c[1], Config::kCoefficientBits[D - 1]>();
    log_fused_load_constant<c[2], Config::kCoefficientBits[D - 2]>();
    log_fused_load_constant<c[3], Config::kCoefficientBits[D - 3]>();
    if constexpr (D == 5u)
    {
        log_fused_load_constant<c[4], Config::kCoefficientBits[1]>();
    }
    constexpr std::uint32_t negative_scale = unit_scale ? p_sfpu::LCONST_neg1 : p_sfpu::LREG7;
    if constexpr (!unit_scale)
    {
        log_fused_load_constant<p_sfpu::LREG7, Config::kScaleBits ^ 0x80000000u>();
    }
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG2, 0, kLogFusedHold, 0); // x
    // TT-NN's datacopy keeps -0 and subnormals in DEST; x*1 + 0 sends both to +0, so log is -Inf.
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0x7FC0);                                      // NaN unless x is positive and finite
    TTI_SFPEXEXP(0, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPEXEXP_MOD1_NODEBIAS); // biased b
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);            // x >= 0
    TTI_SFPIADD(0xF01, p_sfpu::LREG1, p_sfpu::LREG1,
                sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0); // b - 255 < 0: x finite
    // After x*1 + 0, zeros and subnormals of either sign are +0, the only input left open with x == 0.
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0xFF80); // log(0) = -Inf
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPSETEXP(127, p_sfpu::LREG2, p_sfpu::LREG2, 1);                                // m
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);                                     // 255 - b
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG2, 0); // u = m - 1
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);                                       // u-gap filler
    TTI_SFPMAD(c[0], p_sfpu::LREG2, c[1], p_sfpu::LREG3, 0);                            // h = c[D]*u + c[D-1]
    TTI_SFPADDI(0xC300, p_sfpu::LREG1, 0);                                              // -e = 127 - b, exact; h-gap filler
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, c[2], p_sfpu::LREG3, 0);                   // + c[D-2]
    TTI_SFPMUL(p_sfpu::LREG1, negative_scale, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);      // e*scale; gap filler
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, c[3], p_sfpu::LREG3, 0);                   // + c[D-3]
    if constexpr (D == 5u)
    {
        TTI_SFPNOP;
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, c[4], p_sfpu::LREG3, 0); // + c1
    }
    TTI_SFPNOP;
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3, 0); // y = h*u + e*scale
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LREG3,
                      sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B); // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kLogFusedAdvance, 0);     // dest += 2
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16
