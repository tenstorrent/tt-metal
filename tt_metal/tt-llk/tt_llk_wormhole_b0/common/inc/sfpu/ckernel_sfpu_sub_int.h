// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _sub_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // Operand A is input1 (int32/uint16)
    // Operand B is input2 (int32/uint16)
    // Output is int32/uint16
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Use '12' if Dest is in sign-magnitude format and '4' for 2's complement,
    // because integer add/sub requires 2's complement format in LREGs.
    constexpr InstrModLoadStore sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : INSTRUCTION_MODE;

    // Each Dest tile is 64 rows; sfpi dst_reg[] indexes in stride units (SFP_DESTREG_STRIDE == 2),
    // so 64 raw rows == 32 sfpi stride units.
    constexpr std::uint32_t dst_tile_size = 32;

    // out = in0 - in1. sfpi's `a - b` lowers to the same SFPIADD that 2's-complements the subtrahend,
    // matching the original TTI_SFPIADD(..., imod 6). The load/store DataLayout is chosen so its SFP
    // load/store format byte equals the original InstrModLoadStore value:
    //   INT32 (4) -> I32 (sign-mag<->2's-comp conversion), LO16 (6) -> U16, INT32_2S_COMP (12) -> SM32 (raw).
    constexpr sfpi::DataLayout layout = (sfpload_instr_mod == InstrModLoadStore::LO16)             ? sfpi::DataLayout::U16
                                        : (sfpload_instr_mod == InstrModLoadStore::INT32_2S_COMP)  ? sfpi::DataLayout::SM32
                                                                                                  : sfpi::DataLayout::I32;
    using vType                       = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size].mode<layout>();
        vType b = sfpi::dst_reg[dst_index_in1 * dst_tile_size].mode<layout>();
        sfpi::dst_reg[dst_index_out * dst_tile_size].mode<layout>() = a - b;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
