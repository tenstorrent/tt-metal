// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

// replay slot 0 = num_sfpu_iterations SFPLOADMACROs (load+exp) then the same count of SFPSTOREs
inline constexpr std::uint32_t _exp_loadmacro_replay_len_(const int num_sfpu_iterations)
{
    return static_cast<std::uint32_t>(num_sfpu_iterations) * 2;
}

// Program macro seq 0 (load -> NONLINEAR(EXP), result in the load LREG) and record the per-slice
// stream into replay slot 0. Store is explicit because the fused store only writes in-place and we
// need a separate output slice. Load/store are split into two phases and the LREG rotates (d & 3)
// so the exp has drained before its store. Because the LREG index only rotates over 4 registers,
// this two-phase scheme is only correct for num_sfpu_iterations <= 4; beyond that the phase-1 loads
// would overwrite an LREG whose exp result the phase-2 store has not yet drained.
// SFPLOADMACRO addr is the [10:1] field, hence >> 1.
inline void _exp_init_loadmacro_(const std::uint32_t load_base_addr, const std::uint32_t store_base_addr, const int num_sfpu_iterations)
{
    LLK_ASSERT(num_sfpu_iterations <= 4, "Two-phase exp LOADMACRO recorder only supports num_sfpu_iterations <= 4 (LREG index rotates d & 3)");

    // capture NONLINEAR(EXP) into instr reg 4 (VD=0xC is the backdoor)
    TTI_SFPNONLINEAR(0x0 /* VC */, 0xC /* VD */, p_sfpnonlinear::EXP_MODE);

    // seq reg 0: SIMPLE=0x04 (instr 4, delay 0, no staging), MAD/ROUND/STORE unused
    constexpr std::uint32_t simple_bits = 0x04;
    TTI_SFPLOADI(0x0, 0xA, simple_bits); // [MAD | SIMPLE]
    TTI_SFPLOADI(0x0, 0x8, 0x0000);      // [STORE | ROUND]
    TTI_SFPCONFIG(0x0000, 0x4, 0x0);

    load_replay_buf(
        0,
        _exp_loadmacro_replay_len_(num_sfpu_iterations),
        false,
        0,
        0,
        [load_base_addr, store_base_addr, num_sfpu_iterations]
        {
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                TT_SFPLOADMACRO(0, d & 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, (load_base_addr + (d << 1)) >> 1, 0);
            }
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                TT_SFPSTORE(d & 3, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, store_base_addr + (d << 1));
            }
        });
}

// Calculates EXP for number of rows of output SFPU ops (Quasar = 2 rows)
template <bool APPROXIMATION_MODE>
inline void _calculate_exp_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::EXP_MODE); // Read value from lreg[0], approximate recip, load back into lreg[1]

    // Store from lreg[1] into dest register
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 0);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_exp_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        _calculate_exp_sfp_rows_<APPROXIMATION_MODE>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>(); // does the dest_reg++ (increments by 2 rows)
    }
}

} // namespace sfpu
} // namespace ckernel
