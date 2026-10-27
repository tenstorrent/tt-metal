// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"
#include "register_layout.h"
#include "tensix.h"

namespace ckernel
{
/** @brief Legacy software tracker of the selected state-CFG bank, shared until its callers migrate to HAL. */
extern std::uint32_t cfg_state_id;
} // namespace ckernel

namespace hal::cfg::detail
{

// This duplicates ckernel::get_cfg_pointer.
// TODO(njokovic) issue #58443: Remove ckernel:: implementation when HAL is applied to all kernels.

/**
 * @brief Return the state-CFG MMIO bank selected by the software state tracker.
 *
 * @return Bank base indexed in 32-bit words, selected using ckernel::cfg_state_id.
 * @note Keep ckernel::cfg_state_id synchronized with the thread's CFG_STATE_ID when changing banks.
 */
inline volatile std::uint32_t tt_reg_ptr* state_cfg_bank()
{
    const std::uint32_t bank_offset = (ckernel::cfg_state_id == 0) ? 0u : StateCfgWordCount;
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE) + bank_offset;
}

} // namespace hal::cfg::detail
