// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Selected product bodies. Caller owns initialization and SFPU start/done.
#include <cstdint>

namespace sfpi
{
template <class Config, std::uint32_t Hold, std::uint32_t Advance>
inline void product_bounded_bh()
{
    constexpr std::uint32_t kMultBits                  = __builtin_bit_cast(std::uint32_t, (float)Config::kMultiplier);
    constexpr std::uint32_t kBiasBits                  = __builtin_bit_cast(std::uint32_t, (float)Config::kBias);
    constexpr std::uint32_t kC0Bits                    = __builtin_bit_cast(std::uint32_t, Config::kCoefficients[0]);
    constexpr std::uint32_t kNegSlopeBits              = __builtin_bit_cast(std::uint32_t, (float)Config::kNegativeSlope);
    constexpr std::uint32_t kNegativePositiveScaleBits = __builtin_bit_cast(std::uint32_t, -(float)Config::kPositiveScale);

    auto restore_cross_row_pins = [&]()
    {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kMultBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kMultBits & 0xffffu));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, (kBiasBits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, (kC0Bits >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, (kC0Bits & 0xffffu));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, Config::kCoordinateUpperBf16);
    };
    restore_cross_row_pins();

    TTI_REPLAY(0, 32, 1, 1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 0); // 1: x
    TTI_SFPMAD(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0,
               0); // 2: x*log2(e)+scaled bias
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER,
                 (kNegSlopeBits >> 16)); // 3: producer gap
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0,
                9); // 4: max(xlog2, 0)
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER,
                 (kNegSlopeBits & 0xffffu)); // 5: lower-clamp retirement gap
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0,
                1); // 6: min(xlog2, typed right-bound coordinate)
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER,
                 (kNegativePositiveScaleBits >> 16));                   // 7: clamp retirement gap
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 0); // 8
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0); // 9
    TTI_SFPSHFT(0, 4, 0, 0);                                            // 10
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5,
                 sfpi::SFPEXMAN_MOD1_PAD9); // 11
    TTI_SFPCAST(5, 5, 0);                   // 12
    TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG4,
               0); // 13: P2 head
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER,
                 (kNegativePositiveScaleBits & 0xffffu)); // 14: Horner gap
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG4,
               0); // 15: P2 tail
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB,
                 0x4380);      // 16: 2*scale == 256, also Horner gap
    TTI_SFPSETEXP(0, 4, 0, 2); // 17: scaled exp(x) in L0
    TTI_SFPNOP;                // 18: SETEXP result retirement
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5,
               0); // 19: s^2
    TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3,
               0); // 20: x*s, also square retirement gap
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5,
               0); // 21: s^2 + 256*s
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        0);                                                              // 22: negative numerator, cross-term retirement gap
    TTI_SFPADDI(0x4700, ckernel::p_sfpu::LREG5, 0);                      // 23: +32768
    TTI_SFPNOP;                                                          // 24: denominator retirement
    TTI_SFPARECIP(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, 0); // 25
    TTI_SFPNOP;                                                          // 26: reciprocal seed retirement
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG5,
               2);                                 // 27: reciprocal Newton residual
    TTI_SFPNOP;                                    // 28
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, 0); // 29
    TTI_SFPMAD(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4,
               3);            // 30: refined reciprocal
    TTI_SFPENCC(3, 0, 0, 10); // 31
    TTI_SFPMUL(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG6,
               0); // 32: negative numerator / denominator

    auto same_row_suffix = [&]()
    {
        TTI_SFPMAD(
            ckernel::p_sfpu::LREG7,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG5,
            0); // positive factor, also negative-ratio retirement gap
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG6,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG3,
            0); // negative result, also positive-factor retirement gap
        TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5,
                   0);                                 // positive result
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, 0); // x < 0
        TTI_SFPMOV(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        // Physical BF16 sign/exponent and mantissa are disjoint. After
        // XOR, the second predicate excludes exact -Inf without a new mask.
        TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, Config::kRawNegativeNanWord);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG5, 0, Advance, 0);
        restore_cross_row_pins();
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (int row = 1; row < 32; ++row)
    {
        TTI_REPLAY(0, 32, 0, 0);
        same_row_suffix();
    }
}
} // namespace sfpi
