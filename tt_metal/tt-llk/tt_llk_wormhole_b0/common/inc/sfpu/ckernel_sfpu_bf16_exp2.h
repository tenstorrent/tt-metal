// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_bf16_exp2_paired.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16
{
constexpr std::uint32_t kExpHold = ADDR_MOD_3, kExpAdvance = ADDR_MOD_2;

template <typename Config>
inline void exp2_clamp_pins()
{
    // Clamp operands are destroyed by SWAP; SFPLOADI must execute per replay.
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);
}

template <typename Config, int Iterations = 8>
inline void calculate_exp2()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    static_assert(Config::kBodySlots == 2 * (10 + Config::kDegree) + 4);
    static_assert(Config::kDegree != 3 || Config::kScaledCoefficientBits[0] == 0x3f800000u);
    constexpr std::uint32_t replay_slots = Config::kBodySlots;
    constexpr std::uint32_t last         = Config::kScaledCoefficientBits[Config::kDegree == 3 ? 1 : 0];
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, last >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, last & 0xffffu);
    exp2_clamp_pins<Config>();
    // The NaN fix is issued between two recorded parts: the whole body would
    // not fit the 32-slot replay buffer.
    [[maybe_unused]] constexpr std::uint32_t scale_slots = 4; // read only with the -NaN terminal
    TTI_REPLAY(0, replay_slots, 1, 1);
    sfpi::exp2_paired_body<Config::kDegree, true, kExpHold>();
    // The complete primitive already owns the target special classes.
    // ADDR_MOD_6 keeps stock's dest += 2, which other unary SFPU ops read, so each store advances.
    TTI_SFPSTORE(p_sfpu::LREG0, 0, kExpAdvance, 0);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, kExpAdvance, 0);
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair)
    {
        exp2_clamp_pins<Config>();
        TTI_REPLAY(0, replay_slots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16
