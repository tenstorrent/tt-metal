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

template <class Config, std::uint32_t Hold>
inline void product_symmetric_wh()
{
    static_assert(Config::kDomainActionCount == 3);
    static_assert(sizeof(Config::kCoefficients) / sizeof(float) == 3);
    constexpr std::uint32_t origin_bf16 = ((Config::kOriginEncoded & 0xffu) << 7) | ((Config::kOriginEncoded >> 8) & 0x7fu);
    static_assert(origin_bf16 > 0 && origin_bf16 < 2048);
    constexpr std::uint32_t middle     = __builtin_bit_cast(std::uint32_t, Config::kCoefficients[1] * 0x1p-23f);
    constexpr std::uint32_t top        = __builtin_bit_cast(std::uint32_t, Config::kCoefficients[2] * 0x1p-46f);
    constexpr std::uint32_t bottom     = __builtin_bit_cast(std::uint32_t, Config::kCoefficients[0]);
    constexpr std::uint32_t multiplier = __builtin_bit_cast(std::uint32_t, (float)Config::kMultiplier);
    TTI_SFPLOADI(6, SFPLOADI_MOD0_LOWER, multiplier & 65535);
    TTI_SFPLOADI(6, SFPLOADI_MOD0_UPPER, multiplier >> 16);
    // Upstream LLK blocks several tiles in one DEST half, so the three coefficients
    // are immediates rather than DEST rows. Their loads do not fit the 32-slot replay
    // body, so it is recorded in two parts around one MAD and the last coefficient load.
    auto row_lead = [&]()
    {
        TTI_SFPSETSGN(0, 2, 0, 1);
        TTI_SFPLOADI(1, SFPLOADI_MOD0_FLOATB, Config::kBiasBf16);
        TTI_SFPMAD(0, 6, 1, 0, 0);
        TTI_SFPLOADI(4, SFPLOADI_MOD0_LOWER, middle & 65535);
        TTI_SFPLOADI(4, SFPLOADI_MOD0_UPPER, middle >> 16);
        TTI_SFPEXEXP(0, 0, 3, 0);
        TTI_SFPEXMAN(0, 0, 1, 0);
        TTI_SFPSHFT(0, 3, 1, 0);
        TTI_SFPEXEXP(0, 1, 0, 1);
        TTI_SFPEXMAN(0, 1, 1, 1);
        TTI_SFPCAST(1, 1, 0);
        TTI_SFPLOADI(5, SFPLOADI_MOD0_LOWER, top & 65535);
        TTI_SFPLOADI(5, SFPLOADI_MOD0_UPPER, top >> 16);
    };
    auto row_tail = [&]()
    {
        TTI_SFPMAD(3, 1, 7, 1, 0);
        TTI_SFPLOADI(4, SFPLOADI_MOD0_FLOATB, __builtin_bit_cast(std::uint32_t, Config::kDomainActions[2].bound) >> 16);
        TTI_SFPSETEXP(0, 1, 0, 0);
        TTI_SFPLOADI(1, SFPLOADI_MOD0_FLOATB, __builtin_bit_cast(std::uint32_t, -Config::kScale) >> 16);
        TTI_SFPMAD(0, 1, 11, 1, 0);
        TTI_SFPMAD(11, 2, 4, 4, 0);
        TTI_SFPMAD(12, 1, 13, 3, 0);
        TTI_SFPLOADI(5, SFPLOADI_MOD0_FLOATB, __builtin_bit_cast(std::uint32_t, Config::kDomainActions[1].bound) >> 16);
        TTI_SFPMAD(3, 1, 14, 3, 0);
        TTI_SFPMAD(11, 2, 5, 5, 0);
        TTI_SFPMAD(1, 3, 10, 1, 0);
        TTI_SFPSETSGN(0, 2, 7, 1);
        TTI_SFPMAD(3, 1, 3, 1, 0);
        TTI_SFPMUL(7, 0, 9, 0, 0);
        TTI_SFPMULI(__builtin_bit_cast(std::uint32_t, -2.0f * Config::kScale) >> 16, 1, 0);
        TTI_SFPADD(10, 7, 2, 3, 0);
        TTI_SFPSHFT((-16 & 4095), 7, 7, 1);
        TTI_SFPMAD(0, 1, 3, 0, 0);
        TTI_SFPIADD((-origin_bf16 & 4095), 7, 7, SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_NONE);
    };
#pragma GCC unroll 32
    for (int row = 0; row < 32; ++row)
    {
        TTI_SFPLOAD(2, 0, Hold, row * 2);
        if (row == 0)
        {
            ::ckernel::sfpu::bf16::silu::record_row<0, 13>(row_lead);
        }
        else
        {
            ::ckernel::sfpu::bf16::silu::replay_row<0, 13>(row_lead);
        }
        TTI_SFPMAD(5, 1, 4, 3, 0);
        TTI_SFPLOADI(7, SFPLOADI_MOD0_LOWER, bottom & 65535);
        TTI_SFPLOADI(7, SFPLOADI_MOD0_UPPER, bottom >> 16);
        if (row == 0)
        {
            ::ckernel::sfpu::bf16::silu::record_row<13, 19>(row_tail);
        }
        else
        {
            ::ckernel::sfpu::bf16::silu::replay_row<13, 19>(row_tail);
        }
        TTI_SFPSWAP(0, 0, 2, 1);
        TTI_SFPMULI(0x3f00, 0, 0);
        TTI_SFPLOAD(1, 0, Hold, row * 2);
        TTI_SFPSETCC(0, 7, 0, SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSETMAN(0, 1, 0, 1);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, 4, 0, SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPMOV(0, 1, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, 5, 0, SFPSETCC_MOD1_LREG_GTE0);
        TTI_SFPMUL(1, 9, 9, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 0, 0, 0, SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPSTORE(0, 0, Hold, row * 2);
    }
}

} // namespace ckernel::sfpu::ttpoly::silu
