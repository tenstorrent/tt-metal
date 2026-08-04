// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../access_types.h"

namespace hal::cfg::detail
{

inline constexpr std::uint32_t DynamicGprIndex = 0xffffffffu;

/**
 * Internal operand returned by the public gpr() factories.
 *
 * The index template argument keeps compile-time and runtime operands distinct,
 * allowing read()/write() to select TTI_* or TT_* with if constexpr.
 */
template <std::uint32_t Index, GprTransferSize Size, WrcfgCompletion Completion>
struct GprOperand
{
    static constexpr std::uint32_t index        = Index;
    static constexpr GprTransferSize size       = Size;
    static constexpr WrcfgCompletion completion = Completion;
};

template <GprTransferSize Size, WrcfgCompletion Completion>
struct GprOperand<DynamicGprIndex, Size, Completion>
{
    std::uint32_t index;
    static constexpr GprTransferSize size       = Size;
    static constexpr WrcfgCompletion completion = Completion;
};

} // namespace hal::cfg::detail
