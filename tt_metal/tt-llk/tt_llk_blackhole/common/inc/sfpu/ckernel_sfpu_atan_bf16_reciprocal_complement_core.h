// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Selected reciprocal coordinate and reconstruction. Callers retain terminals,
// coefficient ownership, rounding and traversal.
namespace ckernel::sfpu::ttpoly::atan
{
using namespace ::sfpi;
using ::sfpi::DataLayout;
// The SFPLOADIs a Held row adds for its two leading coefficients.
constexpr unsigned reciprocal_complement_held_slots = 4u;

// Below degree eight, rung J feeds coefficient K = kDegree - 2 - J: the first three
// from LREG7/6/5, the next from LREG3 loaded in the gaps of rungs 0 and 1, a fifth
// through LREG3 again in its own rung, and a zero constant term from LCONST_0.
template <class Config, unsigned J>
constexpr unsigned reciprocal_complement_short_addend()
{
    if constexpr (Config::kDegree - 2u == J)
    {
        return 9; // admission requires c0 == 0
    }
    else if constexpr (J < 3)
    {
        return 7u - J;
    }
    else
    {
        return 3;
    }
}

template <class Config, unsigned J>
inline void reciprocal_complement_short_rung()
{
    constexpr unsigned kLateLoad = Config::kDegree - 5u;
    if constexpr ((J == 0 || J == 1) && Config::kDegree >= 6)
    {
        constexpr unsigned word = Config::kCoefficientBits[kLateLoad];
        if constexpr (J == 0)
        {
            TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, word >> 16);
        }
        else
        {
            TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, word & 0xffffu);
        }
    }
    else if constexpr (J == 4 && Config::kDegree == 7)
    {
        TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[1] >> 16);
        TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[1] & 0xffffu);
    }
    else
    {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(2, 1, (reciprocal_complement_short_addend<Config, J>()), 2, 0);
    if constexpr (J + 2u < Config::kDegree)
    {
        reciprocal_complement_short_rung<Config, J + 1u>();
    }
}

// LoadMacro false issues what the load macro schedules (init_reciprocal_complement) as plain instructions,
// for builds without SFPLOADMACRO.
// Held false reads the two leading coefficients from vConstFloatPrgm1/Prgm2, which the caller's init
// programs; Held true loads them in the row (reciprocal_complement_held_slots more slots), into LREG2 and
// LREG3, both dead between the reduction's swap and the first Horner MAD.
template <class Config, unsigned Hold, unsigned Advance, bool LoadMacro = true, bool Held = false>
inline void reciprocal_complement_body()
{
    static_assert(!Held || Config::kDegree != 8, "the degree-eight row has no free replay slot");
    TTI_SFPLOAD(0, 0, Hold, 0);
    TTI_SFPNOP;
    TTI_SFPSETSGN(0, 0, 1, 1);
    TTI_SFPARECIP(0, 1, 3, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, 1, ckernel::p_sfpu::LCONST_1, 0, 0);
    TTI_SFPMAD(1, 3, 12, 2, 2);
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 2, 0, 0);
    TTI_SFPMAD(2, 3, ckernel::p_sfpu::LCONST_0, 3, 3);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSWAP(0, 3, 1, 1);
    if constexpr (Held)
    {
        TTI_SFPLOADI(2, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree] >> 16);
        TTI_SFPLOADI(2, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree] & 0xffffu);
        TTI_SFPLOADI(3, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[Config::kDegree - 1] >> 16);
        TTI_SFPLOADI(3, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[Config::kDegree - 1] & 0xffffu);
        TTI_SFPMAD(2, 1, 3, 2, 0);
    }
    else
    {
        TTI_SFPMAD(13, 1, 14, 2, 0);
    }
    reciprocal_complement_short_rung<Config, 0>();
    TTI_SFPSETCC(0, 0, 0, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, 2, 4, 2, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    if constexpr (LoadMacro)
    {
        TTI_SFPLOADMACRO(3, 0, Advance, 0);
    }
    else
    {
        // x's sign onto the result, Nearest to BF16, stored over x.
        TTI_SFPLOAD(3, 0, Hold, 0);
        TTI_SFPNOP;
        TTI_SFPSETSGN(0, 2, 3, 0);
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, 3, 3, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(3, 0, Advance, 0);
    }
}
} // namespace ckernel::sfpu::ttpoly::atan
