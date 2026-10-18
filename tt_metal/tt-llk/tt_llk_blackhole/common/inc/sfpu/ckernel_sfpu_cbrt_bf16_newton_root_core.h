// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Selected Newton rows. Callers own replay, init and the destination walk.
#include <cstdint>

namespace ckernel::sfpu::ttpoly::cbrt
{
using namespace ::sfpi;
using ::sfpi::DataLayout;

template <std::uint32_t Hold, std::uint32_t Advance>
__attribute__((always_inline)) inline void newton_cbrt_body()
{
    TTI_SFPLOAD(ckernel::p_sfpu::LREG3, 0, Hold, 0);                  // x
    TTI_SFPNOP;                                                       // replay load-to-consumer gap
    TTI_SFPABS(0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG4, 1); // ax
    TTI_SFPCAST(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
               0); // f = bits*negative_third_256 + magic
    TTI_SFPNOP;    // MAD-to-shift gap
    TTI_SFPSHFT(0x008, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 7);
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG5,
               0); // y*y
    TTI_SFPNOP;    // MUL-to-consumer gap
    TTI_SFPMUL(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG4,
               0); // d = ax*y*y
    TTI_SFPNOP;    // MUL-to-consumer gap
    TTI_SFPMUL(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0,
               0); // c = d*y
    TTI_SFPNOP;    // MUL-to-consumer gap
    TTI_SFPMAD(ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG5,
               0); // C2*c + C1
    TTI_SFPNOP;    // MAD-to-consumer gap
    TTI_SFPMAD(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0,
               0);                                                       // t = c*(C2*c+C1)+C0
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG3, 0); // signed d
    TTI_SFPMUL(ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG0,
               0); // t*t
    TTI_SFPNOP;    // MUL-to-consumer gap
    TTI_SFPMUL(ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG3,
               0); // y = signed d*(t*t)
    TTI_SFPNOP;    // MUL-to-RNE gap
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN, 0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG3, 0, Advance, 0);
}
} // namespace ckernel::sfpu::ttpoly::cbrt
