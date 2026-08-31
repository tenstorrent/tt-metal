// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel
{
namespace sfpu
{

// Squares one pair of rows (Quasar SFPU ops cover 2 rows)
inline void _calculate_square_rows_(const int load_addr, const int store_addr, const std::uint32_t load_sfpmem, const std::uint32_t store_sfpmem)
{
    TT_SFPLOAD(p_sfpu::LREG0, load_sfpmem, ADDR_MOD_7, 0, load_addr);             // load from load_addr into lreg[0]
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // lreg[0] = lreg[0] * lreg[0]
    TT_SFPSTORE(p_sfpu::LREG0, store_sfpmem, ADDR_MOD_7, 0, store_addr);          // store lreg[0] to store_addr
}

// Implements element-wise square: reads num_sfpu_iterations row pairs starting at load_base_addr
// and writes the squares starting at store_base_addr. The addresses select the register file the
// SFPU accesses: Dest (address bit 10 = 0) or SrcS (bit 10 = 1).
// Caller resolves the sfpmem types: Float16 needs an explicit FP16A, sfpmem::DEFAULT never is.
inline void _calculate_square_(
    const int load_base_addr, const int store_base_addr, const int num_sfpu_iterations, const std::uint32_t load_sfpmem, const std::uint32_t store_sfpmem)
{
#pragma GCC unroll 8
    for (int d = 0; d < num_sfpu_iterations; d++)
    {
        _calculate_square_rows_(load_base_addr + (d << 1), store_base_addr + (d << 1), load_sfpmem, store_sfpmem);
    }
}

} // namespace sfpu
} // namespace ckernel
