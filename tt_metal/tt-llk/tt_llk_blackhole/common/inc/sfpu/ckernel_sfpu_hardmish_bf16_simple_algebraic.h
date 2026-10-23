// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// StepEach: Advance steps one row (dest += 2, stock's ADDR_MOD_6), so both stores advance.
#include <cstdint>

namespace ckernel::sfpu::ttpoly::hardmish
{
using namespace ::sfpi;
using ::sfpi::DataLayout;

template <std::uint32_t A, std::uint32_t B, std::uint32_t Hold, std::uint32_t Advance, bool StepEach = false>
inline void simple_finish_pair()
{
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, A, A, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, B, B, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    if constexpr (StepEach)
    {
        TTI_SFPSTORE(A, 0, Advance, 0);
        TTI_SFPSTORE(B, 0, Advance, 0);
    }
    else
    {
        TTI_SFPSTORE(A, 0, Hold, 0);
        TTI_SFPSTORE(B, 0, Advance, 2); // pair complete, dest += 4
    }
}

template <
    std::uint32_t A,
    std::uint32_t B,
    std::uint32_t RAWA,
    std::uint32_t RAWB,
    std::uint32_t MASK,
    std::uint32_t SCRATCH,
    std::uint32_t EqualValue,
    std::uint32_t NonzeroMask,
    std::uint32_t Output,
    std::uint32_t Hold>
inline void simple_raw_terminal_pair()
{
    TTI_SFPLOAD(RAWA, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
    TTI_SFPLOAD(RAWB, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 2);
    TTI_SFPLOADI(MASK, sfpi::SFPLOADI_MOD0_USHORT, EqualValue);
    TTI_SFPXOR(0, MASK, RAWA, 0);
    TTI_SFPXOR(0, MASK, RAWB, 0);
    if constexpr (NonzeroMask == 0u)
    {
        TTI_SFPSETCC(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(A, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPLOADI(B, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
    }
    else
    {
        TTI_SFPAND(MASK, RAWA, SCRATCH, 1);
        TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, RAWA, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(A, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPAND(MASK, RAWB, SCRATCH, 1);
        TTI_SFPSETCC(0, SCRATCH, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETCC(0, RAWB, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TTI_SFPLOADI(B, sfpi::SFPLOADI_MOD0_FLOATB, Output);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <std::uint32_t EqualValue, std::uint32_t Hold, std::uint32_t Advance>
inline void simple_bh_gated_joint_row_tail()
{
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0); // 16
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0xFF80); // 17
    TTI_SFPENCC(0, 0, 0, 0);                                                  // 18
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // 19
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x0000); // 20
    TTI_SFPENCC(0, 0, 0, 0);                                                  // 21
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);           // 22
    TTI_SFPNOP;                                          // 23: RNE/store dependency
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, Advance, 0); // 24
}

template <
    bool OriginHalfTie,
    bool RawTerminal,
    std::uint32_t EqualValue,
    std::uint32_t NonzeroMask,
    std::uint32_t Output,
    std::uint32_t Hold,
    std::uint32_t Advance,
    bool StepEach = false>
inline void simple_gated_pair()
{
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0); // xA
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold, 2); // xB
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 9);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3, 1);
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPMUL(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3, 0);
    if constexpr (OriginHalfTie)
    {
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);    // |xA|
        TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5, 1);    // |xB|
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1); // *2^126
        TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 1); // *2^126
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0);                         // scaled |xA| - 1.9921875
        TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG5, 0);                         // scaled |xB| - 1.9921875
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG3, 1);
        TTI_SFPENCC(0, 0, 0, 0);
    }
    if constexpr (RawTerminal)
    {
        simple_raw_terminal_pair<
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG3,
            ckernel::p_sfpu::LREG4,
            ckernel::p_sfpu::LREG5,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            EqualValue,
            NonzeroMask,
            Output,
            Hold>();
    }
    simple_finish_pair<ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, Hold, Advance, StepEach>();
}
} // namespace ckernel::sfpu::ttpoly::hardmish
