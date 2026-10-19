// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

// The exponent-ALU total form one row at a time, with every constant in L3..L7.
// The paired body needs eleven registers and reads its multiplier and two
// coefficients from the programmable constants; one row needs three working
// registers, so the constants fit in LREGs and the setup sets nothing.
// Each lane runs the paired body's instructions in its order, so results match it.
namespace ckernel::sfpu::bf16::exp2
{
constexpr std::uint32_t kExpHeldHold = ADDR_MOD_3;

template <std::uint32_t Lreg, std::uint32_t Bits>
inline void exp2_held_load()
{
    TTI_SFPLOADI(Lreg, sfpi::SFPLOADI_MOD0_UPPER, Bits >> 16);
    TTI_SFPLOADI(Lreg, sfpi::SFPLOADI_MOD0_LOWER, Bits & 0xffffu);
}

// L0 is the row, L1 its fraction and L2 the upper clamp, then the accumulator.
// SFPMAD and SFPSWAP results are ready two cycles after issue; WH does not stall for
// an SFPMAD result and BH would stall that cycle anyway, so each reader waits an SFPNOP.
inline void exp2_held_scale()
{
    TTI_SFPLOAD(p_sfpu::LREG0, 0, kExpHeldHold, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    // SWAP destroys the clamp operand; loading it here also covers the SFPMAD latency.
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
}

template <std::uint32_t Degree>
inline void exp2_held_reduce()
{
    static_assert(Degree == 2u || Degree == 3u);
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 9);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, 1);
    TTI_SFPNOP;
    TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG7, (Degree == 3) ? p_sfpu::LREG2 : p_sfpu::LREG1, 0);
    TTI_SFPNOP;
    if constexpr (Degree == 3)
    {
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
        TTI_SFPNOP;
    }
    TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, kExpHeldHold, 0);
}

template <typename Config, int Iterations = 8>
inline void calculate_exp2_held()
{
    static_assert(Iterations > 0);
    static_assert(Config::kDegree == 2u || Config::kDegree == 3u);
    static_assert(Config::kBodySlots == 2 * (10 + Config::kDegree) + 4);
    static_assert(Config::kDegree != 3 || Config::kScaledCoefficientBits[0] == 0x3f800000u);
    constexpr std::uint32_t scale_slots  = 3;
    constexpr std::uint32_t replay_slots = scale_slots + 12 + 2 * Config::kDegree;
    exp2_held_load<p_sfpu::LREG3, Config::kMultiplierBits>();
    exp2_held_load<p_sfpu::LREG4, Config::kScaledCoefficientBits[Config::kDegree]>();
    exp2_held_load<p_sfpu::LREG6, Config::kScaledCoefficientBits[Config::kDegree - 1]>();
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
    exp2_held_load<p_sfpu::LREG7, Config::kScaledCoefficientBits[Config::kDegree == 3 ? 1 : 0]>();
    TTI_REPLAY(0, replay_slots, 1, 1);
    exp2_held_scale();
    exp2_held_reduce<Config::kDegree>();
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 4
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, replay_slots, 0, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16::exp2
