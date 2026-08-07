// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_load_config.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_(const float value)
{
    // SFPU microcode
    sfpi::vFloat fill_val = value;

    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = fill_val;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void _calculate_fill_int_(const std::uint32_t value)
{
    // SFPU microcode
    if constexpr (INSTRUCTION_MODE == InstrModLoadStore::INT32)
    {
        // Materialize the 32-bit immediate once, outside the loop
        sfpi::vInt fill_val = value;
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = fill_val;
            sfpi::dst_reg++;
        }
    }
    else if constexpr (INSTRUCTION_MODE == InstrModLoadStore::LO16)
    {
        // Materialize the 16-bit immediate once, outside the loop
        sfpi::vUInt fill_val = static_cast<std::uint16_t>(value & 0xFFFF);
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>() = fill_val;
            sfpi::dst_reg++;
        }
    }
    else
    {
        static_assert(false, "INSTRUCTION_MODE must be one of: INT32, LO16.");
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_bitcast_(const std::uint32_t value_bit_mask)
{
    // SFPU microcode
    sfpi::vFloat fill_val = Converter::as_float(value_bit_mask);

    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = fill_val;
        sfpi::dst_reg++;
    }
}
} // namespace ckernel::sfpu
