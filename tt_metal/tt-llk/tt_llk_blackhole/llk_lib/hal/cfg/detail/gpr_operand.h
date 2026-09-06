// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../../utils/gpr.h"
#include "../access_types.h"

namespace hal::cfg::detail
{

inline constexpr std::uint32_t DynamicGprIndex = hal::detail::DynamicGprIndex;

/**
 * Internal operand returned by the public gpr() factories.
 *
 * The index template argument keeps compile-time and runtime operands distinct,
 * allowing read()/write() to select TTI_* or TT_* with if constexpr.
 */
template <std::uint32_t Index, GprTransferSize Size, WrcfgCompletion Completion>
class GprOperand : public hal::Gpr<Index>
{
public:
    static constexpr std::uint32_t index        = Index;
    static constexpr GprTransferSize size       = Size;
    static constexpr WrcfgCompletion completion = Completion;
};

template <GprTransferSize Size, WrcfgCompletion Completion>
class GprOperand<DynamicGprIndex, Size, Completion> : public hal::Gpr<DynamicGprIndex>
{
public:
    constexpr explicit GprOperand(const std::uint32_t index) : hal::Gpr<DynamicGprIndex> {index}
    {
    }

    static constexpr GprTransferSize size       = Size;
    static constexpr WrcfgCompletion completion = Completion;
};

template <GprTransferSize Size, WrcfgCompletion Completion, std::uint32_t Index>
inline constexpr auto with_cfg_policy(const hal::Gpr<Index> source)
{
    if constexpr (Index == DynamicGprIndex)
    {
        return GprOperand<Index, Size, Completion> {source.index};
    }
    else
    {
        return GprOperand<Index, Size, Completion> {};
    }
}

} // namespace hal::cfg::detail
