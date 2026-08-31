// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"

namespace ckernel
{
namespace sfpu
{
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS                 = SFPU_ITERATIONS,
    DataFormat FMT                 = DataFormat::Int32,
    int INSTRUCTION_MODE           = 0,
    bool SIGN_MAGNITUDE_FORMAT     = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static_assert(FMT == DataFormat::Int32, "Only Int32 currently supported for SFPU integer add on Quasar");

    constexpr bool is_int    = (FMT == DataFormat::Int32);
    constexpr auto instr_mod = is_int ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT; // There is a quasar bug with implied fmts + upk to dest, so we need
                                                                                         // use explicit types for int SFPULOAD/STORE TEN-4674
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset      = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset      = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset      = dst_index_out * tile_stride;

    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod, ADDR_MOD_7, 0, in0_offset + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, in1_offset + (d << 1));

        // Dest layout depends on how operands reached dest:
        //   UNP_DEST / Int32 L1 with 2's-comp tiles → 2's-comp Int32
        //   copy_tile Int8 + fp32_dest_acc FPU → sign-mag Int32 (SIGN_MAGNITUDE_FORMAT=true)
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC); // Sign+Mag -> 2SC
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC); // Sign+Mag-> 2SC
        }

        TTI_SFPIADD(
            0x0,
            p_sfpu::LREG0,
            p_sfpu::LREG1,
            p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC); // SFPIADD needs to explicitly disable CC output since CC exu is enabled by default

        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM); // 2SC -> Sing+Mag
        }

        TT_SFPSTORE(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, out_offset + (d << 1));
    }
}

// Calculates ADD for one pair of rows (Quasar SFPU ops cover 2 rows)
inline void _calculate_add_rows_(
    const int in0_addr, const int in1_addr, const int store_addr, const std::uint32_t load_sfpmem, const std::uint32_t store_sfpmem)
{
    TT_SFPLOAD(p_sfpu::LREG0, load_sfpmem, ADDR_MOD_7, 0, in0_addr);                // load in0 into lreg[0]
    TT_SFPLOAD(p_sfpu::LREG1, load_sfpmem, ADDR_MOD_7, 0, in1_addr);                // load in1 into lreg[1]
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0); // lreg[2] = lreg[0] + lreg[1]
    TT_SFPSTORE(p_sfpu::LREG2, store_sfpmem, ADDR_MOD_7, 0, store_addr);            // store lreg[2] to store_addr
}

// Implements element-wise add: reads num_sfpu_iterations row pairs starting at in0_base_addr and
// in1_base_addr and writes the sums starting at store_base_addr. The addresses select the register
// file the SFPU accesses: Dest (address bit 10 = 0) or SrcS (bit 10 = 1).
// Caller resolves the sfpmem types: Float16 needs an explicit FP16A, sfpmem::DEFAULT never is.
inline void _calculate_add_(
    const int in0_base_addr,
    const int in1_base_addr,
    const int store_base_addr,
    const int num_sfpu_iterations,
    const std::uint32_t load_sfpmem,
    const std::uint32_t store_sfpmem)
{
#pragma GCC unroll 8
    for (int d = 0; d < num_sfpu_iterations; d++)
    {
        _calculate_add_rows_(in0_base_addr + (d << 1), in1_base_addr + (d << 1), store_base_addr + (d << 1), load_sfpmem, store_sfpmem);
    }
}

} // namespace sfpu
} // namespace ckernel
