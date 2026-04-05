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
        _sfpu_load_imm32_(p_sfpu::LREG1, value);
    }
    else if constexpr (INSTRUCTION_MODE == InstrModLoadStore::LO16)
    {
        _sfpu_load_imm16_(p_sfpu::LREG1, value);
    }
    else
    {
        static_assert(false, "INSTRUCTION_MODE must be one of: INT32, LO16.");
    }
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPSTORE(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
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
