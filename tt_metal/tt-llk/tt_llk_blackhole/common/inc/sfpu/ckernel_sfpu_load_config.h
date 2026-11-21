// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

inline void _sfpu_load_imm32_(const std::uint32_t dest, const std::uint32_t val)
{
    TT_SFPLOADI(dest, 10, (val & 0xFFFF));      // insmod == 10 will write the lower bits, and not affect the upper bits;
    TT_SFPLOADI(dest, 8, (val >> 16) & 0xFFFF); // insmod == 8 will write the upper bits, and not affect the lower bits;
}

inline void _sfpu_load_imm16_(const std::uint32_t dest, const std::uint32_t val)
{
    TT_SFPLOADI(dest, 2, val & 0xFFFF); // insmod == 2 will write imm16 value treated as unsigned integer, right justified and padded with zeroes on the MSBs
}

inline void _sfpu_load_config32_(const std::uint32_t config_lreg, const std::uint32_t upper16, const std::uint32_t lower16)
{
    // registers 11 through 14 are programmable "constants" which are shared across all 4 rows
    // They are updated only through the CONFIG path, which uses LREG[0] first and then copies it to the desired register location
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, lower16);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, upper16);
    TTI_SFPCONFIG(0 /* Immediate */, config_lreg /* Index */, sfpi::SFPCONFIG_MOD1_SRC_R0_LREG0);
}

inline void _init_sfpu_config_reg()
{
    TTI_SFPCONFIG(0, 0xF, 1);
}

// LCONST_neg1 (LREG11) is a core-wide constant other SFPU kernels read as -1.0. A kernel that reprograms LREG11
// through SFPCONFIG calls this before it returns.
inline void _restore_lconst_neg1_()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0xBF80); // bf16 0xBF80 << 16 == fp32 -1.0
    TTI_SFPCONFIG(0x5555, /*LREG11=*/11, /*MOD1_IMM16_IS_LANE_MASK=*/8);
}

} // namespace sfpu
} // namespace ckernel
