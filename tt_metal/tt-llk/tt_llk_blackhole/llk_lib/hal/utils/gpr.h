// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace hal
{
namespace detail
{
inline constexpr std::uint32_t DynamicGprIndex = 0xffffffffu;
}

/**
 * @brief Identify one compile-time or runtime-selected Tensix GPR.
 *
 * This type carries only GPR identity. Transfer width, completion policy, byte slicing,
 * and destination semantics belong to the operation consuming it.
 *
 * @tparam Index: Compile-time GPR index, or the internal runtime-index sentinel.
 */
template <std::uint32_t Index>
class Gpr
{
public:
    static constexpr std::uint32_t index = Index;
};

template <>
class Gpr<detail::DynamicGprIndex>
{
public:
    std::uint32_t index;
};

/**
 * @brief Construct a compile-time-indexed Tensix GPR operand.
 *
 * @tparam Index: GPR index validated by the consuming operation.
 */
template <std::uint32_t Index>
inline constexpr auto gpr()
{
    static_assert(Index != detail::DynamicGprIndex, "GPR index is reserved by hal::gpr()");
    return Gpr<Index> {};
}

/**
 * @brief Construct a runtime-indexed Tensix GPR operand.
 *
 * @param index: GPR index validated by the consuming operation.
 */
inline constexpr auto gpr(const std::uint32_t index)
{
    return Gpr<detail::DynamicGprIndex> {index};
}

} // namespace hal
