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
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, DataFormat FMT = DataFormat::Int32, int INSTRUCTION_MODE = 0, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    static_assert(FMT == DataFormat::Int32, "Only Int32 currently supported for SFPU integer add on Quasar");

    constexpr bool is_int    = (FMT == DataFormat::Int32);
    constexpr auto instr_mod = is_int ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT; // There is a quasar bug with implied fmts + upk to dest, so we need use
                                                                                         // use explicit types for int SFPULOAD/STORE TEN-4674

    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));
        TT_SFPLOAD(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));

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

        TT_SFPSTORE(p_sfpu::LREG1, instr_mod, ADDR_MOD_7, 0, dst_index_out + (d << 1));
    }
}

} // namespace sfpu
} // namespace ckernel
