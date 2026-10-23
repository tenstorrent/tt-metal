// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel_sfpu_silu_bf16_replay_window.h"

// Selected product bodies. Caller owns initialization and SFPU start/done.
namespace ckernel::sfpu::ttpoly::silu
{
using namespace ::sfpi;
using ::sfpi::DataLayout;

inline void product_symmetric_load_macro_init()
{
#if !defined(DISABLE_SFPLOADMACRO)
    // Reuse the common raw-ingress RNE/store LOADMACRO carrier.  The
    // macro's load into dead L2 is overwritten by the next replay's
    // absolute-value head; RNE reads literal L0 into staging and the
    // fixed store writes the row the load addressed at issue+3.
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    constexpr std::uint32_t kSymmetricProductLmSequence = 0x13850000u;
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (kSymmetricProductLmSequence >> 16));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, (kSymmetricProductLmSequence & 0xFFFFu));
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
    TTI_SFPNOP;
#endif
}

// Held false reads the P2 head's two coefficients from vConstFloatPrgm1/Prgm2, which the caller's init programs.
// Held true records the row without running it and plays it in two parts around four SFPLOADIs that put them in
// L3 (the multiplier, dead after the row's first MAD) and L6 (that MAD's own destination); the row's tail puts the
// multiplier back in L3.
template <class Config, std::uint32_t Hold, std::uint32_t Advance, bool Held = false>
inline void product_symmetric_bh()
{
    constexpr std::uint32_t kSymMultBits          = __builtin_bit_cast(std::uint32_t, (float)Config::kMultiplier);
    constexpr std::uint32_t kSymC0Bits            = __builtin_bit_cast(std::uint32_t, Config::kCoefficients[0]);
    constexpr std::uint32_t kSymHeadBits          = __builtin_bit_cast(std::uint32_t, Config::kCoefficients[2] * 0x1p-46f);
    constexpr std::uint32_t kSymMiddleBits        = __builtin_bit_cast(std::uint32_t, Config::kCoefficients[1] * 0x1p-23f);
    constexpr std::uint32_t kSymRecordedBodyLenD2 = 23;
    // The P2 head's MAD is the record's 11th slot.
    constexpr std::uint32_t kSymHeadSlot = 10;
    // A replay window shorter than the record holds the lead and the rest's first kSymRestHeadSlots slots.
    constexpr std::uint32_t kSymRestHeadSlots = 6;

    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kSymMultBits >> 16));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kSymMultBits & 0xffffu));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, (kSymC0Bits >> 16));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, (kSymC0Bits & 0xffffu));
    // The record: its lead (slots 0..kSymHeadSlot-1), then the rest from the P2 head's MAD.
    auto row_lead = [&]()
    {
        TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 0); // x
        TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB,
                     Config::kBiasBf16); // exact bias; load hazard filler
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2,
                      1); // |x|; preserved through the replay suffix
        TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG7,
                   0); // fused biased xlog2
        TTI_SFPNOP;
        TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 0);
        TTI_SFPSHFT(0, 5, 0, 0);
        TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, 1);
        TTI_SFPCAST(ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 0);
    };
    auto rest_head = [&]()
    {
        if constexpr (Held)
        {
            TTI_SFPMAD(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG6, 0);
        }
        else
        {
            TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG6, 0);
        }
        TTI_SFPGT(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG7, 8);
        TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPAND(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0, 1);
        TTI_SFPSETEXP(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG0,
                      2); // biased exp polynomial y
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG7,
            0); // x+|x|; L1 keeps raw x through the suffix
    };
    auto rest_tail = [&]()
    {
        TTI_SFPMUL(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG6,
                   0);                                  // |x|*y
        TTI_SFPMULI(0x3780, ckernel::p_sfpu::LREG0, 0); // y*2^-16
        TTI_SFPADD(ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0,
                   0); // 1+y*2^-16
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMULI(0x3800, ckernel::p_sfpu::LREG6, 0); // |x|*y*2^-15
        TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, 2);
        TTI_SFPNOP;
    };
    auto row_head = [&]()
    {
        row_lead();
        rest_head();
    };
    if constexpr (Held)
    {
        ::ckernel::sfpu::bf16::silu::record_row<0, kSymHeadSlot, false>(row_lead);
        ::ckernel::sfpu::bf16::silu::record_split<kSymHeadSlot, kSymRecordedBodyLenD2 - kSymHeadSlot, kSymRestHeadSlots, false>(rest_head, rest_tail);
    }
    else
    {
        ::ckernel::sfpu::bf16::silu::record_split<0, kSymRecordedBodyLenD2, kSymHeadSlot + kSymRestHeadSlots>(row_head, rest_tail);
    }
#pragma GCC unroll 32
    for (int d = 0; d < 32; d++)
    {
        if constexpr (Held)
        {
            ::ckernel::sfpu::bf16::silu::replay_row<0, kSymHeadSlot>(row_lead);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kSymHeadBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kSymHeadBits & 0xffffu));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, (kSymMiddleBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, (kSymMiddleBits & 0xffffu));
            ::ckernel::sfpu::bf16::silu::replay_split<kSymHeadSlot, kSymRecordedBodyLenD2 - kSymHeadSlot, kSymRestHeadSlots>(rest_head, rest_tail);
        }
        else if (d != 0)
        {
            ::ckernel::sfpu::bf16::silu::replay_split<0, kSymRecordedBodyLenD2, kSymHeadSlot + kSymRestHeadSlots>(row_head, rest_tail);
        }
        TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5, 3);
        TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG0,
                   1); // doubled signed result
        // Fill the result-MAD -> exact-half dependency window
        // with the independent typed origin-half discriminator.
        // Scaling |x| by 2^126 makes the top first-binade magnitude
        // ordinary 1.9921875, so the test survives FP32 FTZ arithmetic.
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG5, 1);
        // Clamp the doubled result to raw x before halving.  Besides
        // the ordinary lower bound, this owns the positive 0x0100
        // origin word that cannot survive a direct FP32 half under FTZ.
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMULI(0x3F00, ckernel::p_sfpu::LREG0, 0); // exact half
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG5, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0,
                      1); // raw-signed MIN_NORMAL for the exact half tie
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
                    1); // positive identity cap; negative values pass through
#if defined(DISABLE_SFPLOADMACRO)
        // The macro's rounding (its VD as VB and VD) and its store, issued directly.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, Advance, 0);
#else
        TTI_SFPLOADMACRO(2, 0, Advance, 0);
#endif
        // A caller without a stepping address mode passes Hold, and the row steps DEST itself. The macro's
        // store keeps the address its load computed, whatever moves the counter after it.
        if constexpr (Advance == Hold)
        {
            TTI_INCRWC(0, 2, 0, 0);
        }
        if constexpr (Held)
        {
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (kSymMultBits >> 16));
            TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (kSymMultBits & 0xffffu));
        }
    }
#if !defined(DISABLE_SFPLOADMACRO)
    // The final macro's fixed store fires at issue+3.  Other elements
    // drain under the next replay head; only the last needs padding.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}
} // namespace ckernel::sfpu::ttpoly::silu
