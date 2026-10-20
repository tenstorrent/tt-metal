// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace ckernel::sfpu::ttpoly::expm1
{
using namespace ::sfpi;
using ::sfpi::DataLayout;

template <class Config, unsigned Hold, unsigned Advance, class Enter>
inline void factored_cw_wh(Enter enter)
{
    // The typed gate proves BF16 identity with the ordinary WH graph over all inputs.
    // L4..L7 stay pinned; L12..L14 own the three programmable constants.
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[4])) >> 16);
    TTI_SFPLOADI(4, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[4])) & 0xffffu);
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[3])) >> 16);
    TTI_SFPLOADI(5, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[3])) & 0xffffu);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_UPPER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[2])) >> 16);
    TTI_SFPLOADI(6, sfpi::SFPLOADI_MOD0_LOWER, (__builtin_bit_cast(std::uint32_t, Config::kCoefficients[2])) & 0xffffu);
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_UPPER, (0x4b400000u) >> 16);
    TTI_SFPLOADI(7, sfpi::SFPLOADI_MOD0_LOWER, (0x4b400000u) & 0xffffu);
    enter();
    TTI_REPLAY(0, 27, 1, 1);
    TTI_SFPLOAD(0, 0, Hold, 0);
    TTI_SFPLOADI(2, 0, Config::kLowerBf16);
    TTI_SFPSWAP(0, 2, 0, 9);
    TTI_SFPNOP;
    TTI_SFPLOADI(2, 0, Config::kUpperBf16);
    TTI_SFPSWAP(0, 2, 0, 1);
    TTI_SFPNOP;
    TTI_SFPMAD(0, 12, 7, 1, 0);
    TTI_SFPNOP;
    TTI_SFPMOV(0, 1, 3, 0);
    TTI_SFPMAD(7, 11, 1, 1, 0);
    TTI_SFPSHFT(23, 3, 3, 1);
    TTI_SFPIADD(0, 14, 3, 4);
    TTI_SFPMAD(1, 13, 0, 0, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(0, 0, 9, 1, 0);
    TTI_SFPMAD(4, 0, 5, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(2, 0, 6, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(2, 0, 14, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(2, 1, 0, 0, 0);
    TTI_SFPMAD(14, 11, 3, 1, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(3, 0, 1, 2, 0);
    TTI_SFPSETCC(0, 1, 0, 2);
    auto suffix = []()
    {
        TTI_SFPADD(10, 2, 2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, 0, 0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        TTI_SFPLOAD(3, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold, 0);
        TTI_SFPLOADI(2, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
        TTI_SFPXOR(0, 2, 3, 0);
        TTI_SFPAND(0, 3, 2, 0);
        TTI_SFPSETCC(0, 2, 0, 6);
        TTI_SFPSETCC(0, 3, 0, 2);
        TTI_SFPSETEXP(255, 10, 0, 1);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(0, 0, Advance, 0);
        // A store through the hold mode leaves the DEST counter for one row step.
        if constexpr (Advance == Hold)
        {
            TTI_INCRWC(0, 2, 0, 0);
        }
    };
    suffix();
#pragma GCC unroll 32
    for (std::uint32_t row = 1; row < 32; ++row)
    {
        TTI_REPLAY(0, 27, 0, 0);
        suffix();
    }
}
} // namespace ckernel::sfpu::ttpoly::expm1
