// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

// Index-as-type helpers backing the `forEach` methods of the descriptor tables.
//
// A table index has to survive into a template argument, because the Field it
// selects is handed to `write<>` as a `const Field&` non-type template
// parameter. An ordinary `std::uint32_t` loop variable cannot do that, so the
// index travels as a type instead.

namespace hal::cfg::detail
{

/**
 * @brief An index carried as a type, so passing it by value keeps it a constant expression.
 */
template <std::uint32_t Value>
using CompileTimeIndex = std::integral_constant<std::uint32_t, Value>;

/**
 * @brief Invokes `function(CompileTimeIndex<I>{})` once per index in the sequence.
 */
template <typename Function, std::uint32_t... Indices>
inline constexpr void for_each_index(Function&& function, std::integer_sequence<std::uint32_t, Indices...>)
{
    (static_cast<void>(function(CompileTimeIndex<Indices> {})), ...);
}

/**
 * @brief Invokes `function(CompileTimeIndex<I>{})` for each I in [0, Count), unrolled at compile time.
 *
 * The generic lambda parameter is a @ref CompileTimeIndex, so indexing a table
 * with it still yields a Field usable as a template argument:
 *
 *     Unpacker.forEach([&](auto U) {
 *         write<Access::MMIO, Unpacker[U].Cntx[0].Base, Sec::S0>(base[U]);
 *     });
 */
template <std::uint32_t Count, typename Function>
inline constexpr void for_each_index(Function&& function)
{
    for_each_index(static_cast<Function&&>(function), std::make_integer_sequence<std::uint32_t, Count> {});
}

} // namespace hal::cfg::detail
