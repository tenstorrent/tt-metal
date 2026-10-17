// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16
{
constexpr std::uint32_t kLog1pHold = ADDR_MOD_7, kLog1pAdvance = ADDR_MOD_6;

// The canonical body's values in 27 slots. 2^-k x is (x * -4*2^-k) * -0.25, an exact
// product inside the multiply-add that already forms r, so the integer copy of x goes.
// The raw +Inf/NaN discriminator goes too: those inputs reach the SFPU as +Inf, the
// float path carries it to Inf - Inf, and the BH multiply-add returns +NaN, which
// stores +Inf as the discriminator did. x = -1 (u = 0) takes -Inf before the core and
// leaves it, so u dies with e and the row fits L0..L3 around the constants in L4..L7.
constexpr std::uint32_t kLog1pCompactSlots = 27u;

template <typename Config>
inline void log1p_compact_row()
{
    TTI_SFPLOAD(p_sfpu::LREG2, 0, kLog1pHold, 0);                                    // x
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG1, 0); // u = x + 1
    TTI_SFPLOADI(p_sfpu::LREG0, 0, Config::kQuotientBits >> 16);                     // +Inf unless u >= 0
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPLOADI(p_sfpu::LREG0, 0, 0xFF80); // -Inf at u == 0: x = -1
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(p_sfpu::LREG0, 0, Config::kLowerBits >> 16);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);   // e = bits(u) - bits(0.75)
    TTI_SFPSETMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1); // e = k << 23
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 2);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG3, 6);                                 // s = bits(-4) - e = -4 * 2^-k
    TTI_SFPLOADI(p_sfpu::LREG1, 0, 0xBE80);                                          // -0.25
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);    // x * s, exact
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG3, 0); // t = 2^-k - 1
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 3);                                    // e to sign-magnitude
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, 0);       // r = 2^-k x + t
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);                                    // float(e)
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG3, 0);       // p = c2 r + c1
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);    // r^2
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG3, 0);       // p = p r + c0
    TTI_SFPNOP;
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0); // r^2 p + r
    TTI_SFPNOP;
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG2, p_sfpu::LREG0, 0); // + float(e) ln2 2^-23
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0,
                      sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B); // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG0, 0, kLog1pAdvance, 0);
}

template <typename Config, int Iterations = 8>
inline void calculate_log1p()
{
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kBodySlots == 31u);
    static_assert(Config::kLowerBits == 0x3f400000u && Config::kQuotientBits == 0x7f800000u);
    static_assert(Config::kRawEqual == 0xffu);
    constexpr std::uint32_t c0 = Config::kCoefficientBits[0];
    constexpr std::uint32_t c1 = Config::kCoefficientBits[1], c2 = Config::kCoefficientBits[2];
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, c0 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, c0 & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, c1 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, c1 & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, c2 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, c2 & 0xffffu);
    TTI_SFPLOADI(p_sfpu::LREG6, 0, 0xC080); // -4
    TTI_REPLAY(0, kLog1pCompactSlots, 1, 1);
    log1p_compact_row<Config>();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, kLog1pCompactSlots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16
