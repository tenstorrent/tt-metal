// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_softplus_bf16_abs_exp_correction_core.h"
#include "ckernel_sfpu_softplus_bf16_finite_reciprocal.h"
#include "ckernel_sfpu_softplus_bf16_min_max.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::softplus
{
template <typename Config, int Iterations = 32>
inline void calculate_abs_exp_correction()
{
    static_assert(Iterations == 32, "parked correction coefficients require the complete tile");
    {
        ttpoly::softplus::abs_exp_residual_pins<Config>();
        constexpr std::uint32_t slots = 31u + (__builtin_bit_cast(std::uint32_t, Config::kNumerator[0]) == 0x3f800000u ? 0u : 1u);
        load_replay_buf<Exec>(
            0,
            slots,
            [&]
            {
                ttpoly::softplus::abs_exp_residual_core<Config, ADDR_MOD_7, true>();
                // Rows are addressed through ADDR_MOD_7 alone; the DEST counter steps one row after each store.
                ttpoly::softplus::abs_exp_residual_suffix<Config, ADDR_MOD_7>();
                TTI_INCRWC(0, 2, 0, 0);
            });
#pragma GCC unroll 32
        for (int row = 1; row < Iterations; ++row)
        {
            TTI_REPLAY(0, slots, 0, 0);
            ttpoly::softplus::abs_exp_residual_suffix<Config, ADDR_MOD_7>();
            TTI_INCRWC(0, 2, 0, 0);
        }
        return;
    }
    ttpoly::softplus::abs_exp_park_coefficients<Config>();
    auto exp = [](sfpi::vFloat x)
    {
        constexpr bool split_scale_bias = !Config::kSquareDecay;
        x                               = sfpi::setsgn(x, 0);
        return ttpoly::softplus::correction_exp_leaf<Config::kExpDegree, false, false, false, false, true, split_scale_bias>(
            x, Config::kMultiplier, 127.0f, Config::kExpCoefficients);
    };
    auto reciprocal = [](sfpi::vFloat x)
    {
        auto iteration = [](sfpi::vFloat v) { return sfpu_reciprocal_iter<1>(v); };
        return ttpoly::softplus::correction_finite_reciprocal<decltype(iteration), true>(x, iteration);
    };
    for (int row = 0; row < Iterations; ++row)
    {
        sfpi::vFloat raw = sfpi::dst_reg[row];
        sfpi::vFloat result;
        result             = ttpoly::softplus::abs_residual_correction<3, 0, Config>(raw, exp, reciprocal);
        result             = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = result;
    }
}
} // namespace ckernel::sfpu::bf16::softplus
