// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h" // ckernel::cfg_state_id, TENSIX_CFG_BASE
#include "register_layout.h"

namespace hal::cfg::detail
{

// This duplicates ckernel::get_cfg_pointer.
// TODO(njokovic): Remove ckernel:: implementation when HAL is applied to all kernels.

/**
 * @brief Base of the state-CFG bank selected by the current CFG_STATE_ID.
 */
inline volatile std::uint32_t tt_reg_ptr* state_cfg_bank()
{
    const std::uint32_t bank_offset = (ckernel::cfg_state_id == 0) ? 0u : StateCfgWordCount;
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE) + bank_offset;
}

} // namespace hal::cfg::detail
