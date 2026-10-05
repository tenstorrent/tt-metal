// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace sfpi
{
template <class Config, unsigned Hold, unsigned Advance, class Enter>
inline void factored_cw_bh(Enter enter)
{
    constexpr std::uint32_t kRoundingBias = Config::kRoundingBiasBits;
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[4])) >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[4])) & 0xffffu);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[3])) >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[3])) & 0xffffu);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[2])) >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[2])) & 0xffffu);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, (kRoundingBias) >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, (kRoundingBias) & 0xffffu);

    enter();
    TTI_REPLAY(0, 27, 1, 1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    // x * 1 + 0 is exact on every finite input and turns either NaN into the
    // canonical +NaN, which the upper clamp sends to +Inf: -NaN keeps the NaN
    // class without a raw-word test.  The lower clamp's SWAP compares sign and
    // magnitude, so an unconverted -NaN would order below -Inf and saturate to -1.
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kLowerBf16);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 9);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kUpperBf16);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 1);
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP;
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    // Keep L7's rounding bias live across all 31 replays.  The selected leaf's
    // exact c1=0.5 is already preloaded in L14, so it is also the scale/bias
    // bit addend.  Overwriting L7 here made only recorded row zero correct;
    // every replay then rounded with 0.5 instead of 12582912.
    // The shift reads the preserved integer copy in L3, not the MAD-family
    // result in L1.  It is therefore the required independent issue between
    // the L1 producer and its later residual-FMA consumer; no NOP belongs here.
    TTI_SFPSHFT(0x017, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 7);
    TTI_SFPIADD(0, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    // The reduced result is already the k==0 default output in L0.  The base
    // calculation below is independent and supplies its MAD hazard gap; the
    // reconstruction consumes L0 directly, so no L2->L0 copy is required.
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPENCC(0, 0, 0, 0);

    auto same_row_suffix = []()
    {
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (std::uint32_t row = 1; row < 32; ++row)
    {
        TTI_REPLAY(0, 27, 0, 0);
        same_row_suffix();
    }
}
} // namespace sfpi
