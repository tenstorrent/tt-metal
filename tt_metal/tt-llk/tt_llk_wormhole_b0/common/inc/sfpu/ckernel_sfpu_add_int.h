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
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // integer add/sub requires 2's complement format in LREGs.
    constexpr InstrModLoadStore sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : INSTRUCTION_MODE;

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    constexpr sfpi::DataLayout layout = (sfpload_instr_mod == InstrModLoadStore::LO16)            ? sfpi::DataLayout::U16
                                        : (sfpload_instr_mod == InstrModLoadStore::INT32_2S_COMP) ? sfpi::DataLayout::SM32
                                                                                                  : sfpi::DataLayout::I32;
    using vType                       = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        vType a                                                          = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
        vType b                                                          = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() = a + b;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
