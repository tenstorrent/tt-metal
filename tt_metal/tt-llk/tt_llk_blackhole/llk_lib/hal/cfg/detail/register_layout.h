// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../registers.h"

namespace hal::cfg::detail
{

// The RISC CREG debug selector exposes the two complete hardware-CFG banks
// first, followed by the three thread-CFG banks. Anchor these spans to the
// final generated descriptors so additions to either register scope cannot
// silently leave the selector arithmetic at an older hard-coded size.
inline constexpr std::uint32_t HardwareCfgWordCount = ChickenBits::sfpu_scbd_disable.addr32(Sec::S0) + 1u;
inline constexpr std::uint32_t ThreadCfgWordCount   = TensixCsrConfig::RawBusyStatus.addr32(Sec::S0) + 1u;

} // namespace hal::cfg::detail
