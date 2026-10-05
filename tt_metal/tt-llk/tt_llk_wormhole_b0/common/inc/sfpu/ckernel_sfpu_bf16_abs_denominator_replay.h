// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Canonical selected replay arithmetic. Callers own replay count and DST lifecycle.
#include <cstdint>

namespace sfpi
{
// Instructions abs_denominator_wh_suffix issues.
template <bool LateRound, bool RawNegativeInfinity = true>
constexpr std::uint32_t abs_denominator_wh_suffix_slots()
{
    return 5u + (LateRound ? 1u : 0u) + (RawNegativeInfinity ? 2u : 0u);
}

template <std::uint32_t Slopes, std::uint32_t Intercepts>
inline void abs_denominator_lut_init()
{
    TTI_SFPLOADI(0, SFPLOADI_MOD0_USHORT, 0x80ff);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(1, 2, Slopes & 0xffff);
    TTI_SFPLOADI(1, 8, Slopes >> 16);
    TTI_SFPLOADI(5, 2, Intercepts & 0xffff);
    TTI_SFPLOADI(5, 8, Intercepts >> 16);
}

// RawNegativeInfinity keeps the raw word to return -Inf for a raw -Inf input;
// without it every exponent-FF input returns +Inf, the stored NaN.
template <std::uint32_t BoundExponent, bool LateRound, std::uint32_t Hold, bool RawNegativeInfinity = true>
inline void abs_denominator_wh_core()
{
    if constexpr (!RawNegativeInfinity)
    {
        static_assert(LateRound, "the encoded-NaN core rounds after its terminals");
        TTI_SFPLOAD(4, 0, Hold, 0);
        TTI_SFPSETSGN(0, 4, 3, 1);
        TTI_SFPADDI(0x3f80, 3, 0);
        TTI_SFPEXEXP(0, 4, 6, SFPEXEXP_MOD1_NODEBIAS);
        TTI_SFPEXEXP(0, 3, 2, 0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LCONST_neg1, 3, 0);
        TTI_SFPIADD(0, 6, 2, 6);
        TTI_SFPLUTFP32(0, 2);
        TTI_SFPSETEXP(0, 4, 2, 0); // the table reads only L1/L5 for |L3| in [1, 2); fills its gap
        TTI_SFPMUL(2, 0, ckernel::p_sfpu::LCONST_0, 4, 0);
        TTI_SFPMAD(3, 0, ckernel::p_sfpu::LCONST_1, 2, 0);
        TTI_SFPIADD((-255) & 0xfff, 6, 7, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
        TTI_SFPMAD(4, 2, 4, 4, 0);
        TTI_SFPIADD((-BoundExponent) & 0xfff, 6, 6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0);
        return;
    }
    TTI_SFPLOAD(4, 0, Hold, 0);
    TTI_SFPLOAD(7, SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
    TTI_SFPSETSGN(0, 4, 3, 1);
    TTI_SFPADDI(0x3f80, 3, 0);
    TTI_SFPEXEXP(0, 4, 6, SFPEXEXP_MOD1_NODEBIAS);
    TTI_SFPEXEXP(0, 3, 2, 0);
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LCONST_neg1, 3, 0);
    TTI_SFPIADD(0, 6, 2, 6);
    TTI_SFPSETEXP(0, 4, 2, 0);
    TTI_SFPLUTFP32(0, 2);
    TTI_SFPXOR(0, 12, 7, 0);
    TTI_SFPMUL(2, 0, ckernel::p_sfpu::LCONST_0, 4, 0);
    TTI_SFPMAD(3, 0, ckernel::p_sfpu::LCONST_1, 2, 0);
    TTI_SFPIADD((-255) & 0xfff, 6, 3, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
    TTI_SFPMAD(4, 2, 4, 4, 0);
    if constexpr (LateRound)
    {
        TTI_SFPIADD((-BoundExponent) & 0xfff, 6, 6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0);
    }
    else
    {
        TTI_SFPIADD((-BoundExponent) & 0xfff, 6, 6, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 4, 4, 4, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSETCC(0, 6, 0, SFPSETCC_MOD1_LREG_GTE0);
    }
}

template <bool LateRound, std::uint32_t Advance, bool RawNegativeInfinity = true>
inline void abs_denominator_wh_suffix()
{
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LCONST_1, 4, 0);
    TTI_SFPSETCC(0, RawNegativeInfinity ? 3 : 7, 0, SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(4, SFPLOADI_MOD0_FLOATB, 0x7f80);
    if constexpr (RawNegativeInfinity)
    {
        TTI_SFPSETCC(0, 7, 0, SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(4, SFPLOADI_MOD0_FLOATB, 0xff80);
    }
    TTI_SFPENCC(0, 0, 0, 0);
    if constexpr (LateRound)
    {
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 4, 4, 4, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }
    TTI_SFPSTORE(4, 0, Advance, 0);
}
} // namespace sfpi
