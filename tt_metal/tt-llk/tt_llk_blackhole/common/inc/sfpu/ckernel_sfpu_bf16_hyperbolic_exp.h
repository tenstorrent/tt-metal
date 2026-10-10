// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_bf16_hyperbolic_exp_core.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16
{
template <typename Config, int Iterations = 32>
inline void calculate_hyperbolic_exp()
{
    static_assert(Iterations == 32 && Config::kDegree == 4);
    static_assert(Config::kBare);
    // The scale is reloaded per row into LREG4: LREG11 is the -1 constant other ops read.
    sfpi::hyperbolic_exp_pins<Config, false>();
    constexpr unsigned body_slots = Config::kOdd ? 29u : 32u;
    TTI_REPLAY(0, body_slots, 1, 1);
    sfpi::hyperbolic_exp_core<ADDR_MOD_7, sfpi::HyperbolicC0<Config>, sfpi::HyperbolicScale<Config>>();
    sfpi::hyperbolic_even_store<ADDR_MOD_6>();
#pragma GCC unroll 32
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, body_slots, 0, 0);
    }
}
} // namespace ckernel::sfpu::bf16
