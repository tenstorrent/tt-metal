// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace hal::cfg::detail
{

template <std::uint32_t Value>
using CompileTimeIndex = std::integral_constant<std::uint32_t, Value>;

template <typename Function, std::uint32_t... Indices>
inline constexpr void for_each_index(Function&& function, std::integer_sequence<std::uint32_t, Indices...>)
{
    (static_cast<void>(function(CompileTimeIndex<Indices> {})), ...);
}

template <std::uint32_t Count, typename Function>
inline constexpr void for_each_index(Function&& function)
{
    for_each_index(static_cast<Function&&>(function), std::make_integer_sequence<std::uint32_t, Count> {});
}

} // namespace hal::cfg::detail
