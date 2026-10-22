// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu::ttpoly::hardswish
{
using namespace ::sfpi;
using ::sfpi::DataLayout;

template <std::uint32_t Output, std::uint32_t Hold, std::uint32_t Advance>
inline void simple_wh_gated_row()
{
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);                             // 0: x
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0); // 1: raw / x-load gap
    // The WH SFPI baseline emits MUL then ADDI, not a contracted MAD.
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0); // 2: q1*x
    TTI_SFPAND(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, 0);                                                 // 3: raw mask / MUL gap
    TTI_SFPADDI(0x3F00, ckernel::p_sfpu::LREG2, 0);                                                                   // 4: +q0, separate rounding
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG1, 0);                                                 // 5: terminal delta / ADD gap
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 9);                                             // 6: max(0, gate)
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG4, 1);                                              // 7: abs(x)
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, 1);                                             // 8: min(gate, 1)
    TTI_SFPDIVP2(0x07E, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG4, 1);                                           // 9: scale abs(x)
    TTI_SFPNOP;                                                                                                       // 10: DIVP2 -> ADDI gap
    TTI_SFPADDI(0xBFFF, ckernel::p_sfpu::LREG4, 0);                                                                   // 11: boundary delta
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG2, 0); // 12: x * gate
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                                         // 13
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, 1);                                              // 14: signed MIN_NORMAL
    TTI_SFPENCC(0, 0, 0, 0);                                                                                          // 15
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                                         // 16
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Output);                                         // 17
    TTI_SFPENCC(0, 0, 0, 0);                                                                                          // 18
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B); // 19
    TTI_SFPNOP;                                          // 20: round/store gap
    TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, Advance, 0); // 21
}
} // namespace ckernel::sfpu::ttpoly::hardswish
