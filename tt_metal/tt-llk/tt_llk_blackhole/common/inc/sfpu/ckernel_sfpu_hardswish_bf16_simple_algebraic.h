// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu::ttpoly::hardswish
{
using namespace ::sfpi;
using ::sfpi::DataLayout;

template <std::uint32_t EqualValue, std::uint32_t Hold, std::uint32_t Advance>
inline void simple_bh_gated_joint_row()
{
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0); // 0: x
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold,
                0); // 1: raw U16 / x-load gap
    TTI_SFPMAD(ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG2,
               0); // 2: gate
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_USHORT,
                 EqualValue);                                               // 3: gate gap / raw anchor
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);   // 4
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, 0);       // 5
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);    // 6
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);   // 7
    TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1); // 8
    // Raw partitioning is independent of L4 and fills the required
    // DIVP2->ADDI gap.
    TTI_SFPAND(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5,
               1);                                  // 9: negative exponent-FF partition delta
    TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0); // 10
    // The independent x*gate product fills the ADDI->SETCC gap.  It is also
    // separated from its gate-clamp producer at slot 7 by three issues.  No
    // CC window is active across either filler.
    TTI_SFPMUL(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2,
               0);                                                            // 11: x*gate
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // 12
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);      // 13
    TTI_SFPENCC(0, 0, 0, 0);                                                  // 14
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // 15
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
} // namespace ckernel::sfpu::ttpoly::hardswish
