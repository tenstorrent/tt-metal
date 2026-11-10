// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

class Converter
{
public:
    static float as_float(std::uint32_t value)
    {
        return __builtin_bit_cast(float, value);
    }

    // The IEEE-754 bits of a float constant, and the two 16-bit halves SFPLOADI takes as immediates.
    // constexpr so a call can be a template argument or fold into a TTI_ encoding.
    static constexpr std::uint32_t as_uint(float value)
    {
        return __builtin_bit_cast(std::uint32_t, value);
    }

    static constexpr std::uint16_t lo16(float value)
    {
        return static_cast<std::uint16_t>(as_uint(value) & 0xFFFFu);
    }

    static constexpr std::uint16_t hi16(float value)
    {
        return static_cast<std::uint16_t>(as_uint(value) >> 16);
    }
};

} // namespace ckernel::sfpu
