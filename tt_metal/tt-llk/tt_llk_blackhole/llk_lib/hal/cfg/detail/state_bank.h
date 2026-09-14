// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h" // ckernel::cfg_state_id, TENSIX_CFG_BASE
#include "register_layout.h"

namespace hal::cfg::detail
{

// These duplicate ckernel::get_cfg_pointer / cfg_read / cfg_rmw.
// TODO(njokovic): Remove ckernel:: implementation when HAL is applied to all kernels.

/**
 * @brief Base of the state-CFG bank selected by the current CFG_STATE_ID.
 */
inline volatile std::uint32_t tt_reg_ptr* state_cfg_bank()
{
    const std::uint32_t bank_offset = (ckernel::cfg_state_id == 0) ? 0u : StateCfgWordCount;
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE) + bank_offset;
}

/**
 * @brief Read one complete state-CFG word from the active bank.
 */
inline std::uint32_t read_state_word_mmio(const std::uint32_t addr32)
{
    return state_cfg_bank()[addr32];
}

/**
 * @brief Read-modify-write a runtime-masked field of one state-CFG word.
 *
 * Not atomic against the other RISCs sharing the word.
 */
inline void rmw_state_word_mmio(const std::uint32_t addr32, const std::uint32_t shamt, const std::uint32_t mask, const std::uint32_t value)
{
    volatile std::uint32_t* tt_reg_ptr cfg = state_cfg_bank();

    const std::uint32_t old_value = cfg[addr32];
    cfg[addr32]                   = (old_value & ~mask) | ((value << shamt) & mask);
}

} // namespace hal::cfg::detail
