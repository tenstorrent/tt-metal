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

    if constexpr (SIGN_MAGNITUDE_FORMAT)
    {
        // Sign-magnitude operands must be converted to 2's complement in-LREG before/after the
        // integer add, because on Blackhole the INT32_2S_COMP load/store mode has no effect. sfpi
        // can express this via DataLayout::SM32 (which emits smag_to_int/int_to_smag in software),
        // but that generic conversion measures ~35% slower on Blackhole than this raw hand-scheduled
        // path (~498 vs ~368 cyc/tile at MATH_ISOLATE, and #pragma GCC unroll 0 does not help), so
        // the sign-magnitude path stays raw. The 2's-complement branch below is sfpi.
        constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

        // size of each tile in Dest is 64 rows
        constexpr std::uint32_t dst_tile_size = 64;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            // operand B
            TT_SFPLOAD(p_sfpu::LREG0 /*lreg*/, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in1 * dst_tile_size /*dest_reg_addr */);
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG2, INSTR_MOD_CAST);

            // operand A
            TT_SFPLOAD(p_sfpu::LREG1 /*lreg*/, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
            apply_sign_magnitude_conversion(p_sfpu::LREG1, p_sfpu::LREG2, INSTR_MOD_CAST);

            // Set instruction modifier to 6 to get B's 2's complement
            TTI_SFPIADD(0 /*imm*/, p_sfpu::LREG1 /*lreg_c*/, p_sfpu::LREG0 /*lreg_dest*/, 6 /*imod*/);

            // LREG_0 -> dest
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG1, INSTR_MOD_CAST);
            TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_out * dst_tile_size);
            sfpi::dst_reg++;
        }
    }
    else
    {
        // 2's-complement integer subtract expressed in sfpi. `a - b` lowers to the same single
        // SFPIADD (2's-complement of B) the raw path used, but the loads/store and the dst walk are
        // left to the compiler, dropping the per-iteration bookkeeping. Pick the DataLayout whose
        // load/store format byte matches the original InstrModLoadStore (LO16->U16, INT32/2S_COMP->I32;
        // on Blackhole INT32_2S_COMP loads the same raw bits as INT32).
        constexpr sfpi::DataLayout layout = (INSTRUCTION_MODE == InstrModLoadStore::LO16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::I32;
        using vType                       = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            vType a                                                          = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
            vType b                                                          = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() = a - b;
            sfpi::dst_reg++;
        }
    }
}

} // namespace sfpu
} // namespace ckernel
