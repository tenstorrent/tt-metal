// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_cbrt_bf16_newton_root_core.h"
#include "ckernel_sfpu_cbrt_bf16_replay_window.h"
#include "sfpi.h"

// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::bf16::cbrt
{
template <typename Config, int Iterations = 8>
inline void calculate_newton_root();
}

namespace ckernel::sfpu::bf16::cbrt
{
template <typename Config, int Iterations>
inline void calculate_newton_root()
{
    static_assert(Config::kKind == 2);
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kBodySlots == 23u);
    {
        constexpr float negative_inv_n_scaled = (-1.0f / 3.0f) / 256.0f;
        constexpr float magic                 = ((float)Config::kMagic) / 256.0f + 8388608.0f;
        constexpr std::uint32_t scale_bits    = __builtin_bit_cast(std::uint32_t, negative_inv_n_scaled);
        constexpr std::uint32_t magic_bits    = __builtin_bit_cast(std::uint32_t, magic);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, scale_bits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, scale_bits & 0xffffu);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, magic_bits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, magic_bits & 0xffffu);
    }
    {
        auto body = [&] { ttpoly::cbrt::newton_cbrt_body<ADDR_MOD_7, ADDR_MOD_7>(); };
        record_row<0, Config::kBodySlots>(body);
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int row = 1; row < Iterations; ++row)
        {
            replay_row<0, Config::kBodySlots>(body);
            TTI_INCRWC(0, 2, 0, 0);
        }
    }
}
} // namespace ckernel::sfpu::bf16::cbrt
