// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../registers.h"
#include "cfg_defines.h"

namespace hal::cfg::detail
{

// Layout of the memory at TENSIX_CFG_BASE:
//
//     uint32_t Config[2][CFG_STATE_SIZE * 4];
//     uint32_t ConfigDualWrite[CFG_STATE_SIZE * 4];
//     struct {uint16_t Value, Padding;} ThreadConfig[3][THD_STATE_SIZE];
inline constexpr std::uint32_t StateCfgWordCount    = CFG_STATE_SIZE * 4;
inline constexpr std::uint32_t StateCfgBankCount    = 2;
inline constexpr std::uint32_t ConfigDualWriteWords = StateCfgWordCount;
inline constexpr std::uint32_t ThreadCfgWordCount   = THD_STATE_SIZE;

inline constexpr std::uint32_t ThreadCfgBase = StateCfgBankCount * StateCfgWordCount + ConfigDualWriteWords;

static_assert(ChickenBits::sfpu_scbd_disable.addr32(Sec::S0) < StateCfgWordCount, "state CFG descriptor lies outside a hardware CFG bank");
static_assert(TensixCsrConfig::RawBusyStatus.addr32(Sec::S0) < ThreadCfgWordCount, "thread CFG descriptor lies outside a hardware thread bank");

} // namespace hal::cfg::detail
