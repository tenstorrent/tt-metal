// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cfg_defines.h"
#include "ckernel.h"

using namespace ckernel;

/**
 * @brief Configures the SFPU hardware by setting ALU_ACC_CTRL_SFPU_Fp32_enabled for the SFPU thread
 *
 * @tparam EN_32BIT_DEST: Set to true to use Dest in 32bit mode (Float32/Int32)
 */
template <bool EN_32BIT_DEST>
inline void _llk_sfpu_hw_configure_()
{
    // MMIO looks like the obvious, logical way to set this bit, BUT it shares the word with math's Dstacc, Fp32 and INT8 bits,
    // and writing the entire word evicts them all. That is why we use cfg_rmw, which only touches our byte via TT_RMWCIB.
    // cfg[ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32] = EN_32BIT_DEST << ALU_ACC_CTRL_SFPU_Fp32_enabled_SHAMT;
    cfg_rmw(ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW, EN_32BIT_DEST);
}
