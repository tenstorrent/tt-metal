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
 *
 * @tparam Value: Compile-time descriptor-table index.
 */
template <std::uint32_t Value>
using CompileTimeIndex = std::integral_constant<std::uint32_t, Value>;

/**
 * @brief Invoke function once per index, passing each index as a @ref CompileTimeIndex object.
 *
 * The unnamed integer-sequence argument supplies the indices and their iteration order.
 *
 * @tparam Function: Callable accepting a CompileTimeIndex specialization, deduced from function.
 * @tparam Indices: Compile-time index values in invocation order.
 * @param function: Callable to invoke for every index. Return values are discarded.
 */
template <typename Function, std::uint32_t... Indices>
inline constexpr void for_each_index(Function&& function, std::integer_sequence<std::uint32_t, Indices...>)
{
    (static_cast<void>(function(CompileTimeIndex<Indices> {})), ...);
}

/**
 * @brief Invoke function for each index in [0, Count), unrolled at compile time.
 *
 * The generic lambda parameter is a @ref CompileTimeIndex, so indexing a table
 * with it still yields a Field usable as a template argument:
 *
 *     Unpacker.forEach([&](auto U) {
 *         write<Access::MMIO, Unpacker[U].Cntx[0].Base, Sec::S0>(base[U]);
 *     });
 *
 * @tparam Count: Number of indices, starting at zero. Zero invokes nothing.
 * @tparam Function: Callable accepting a CompileTimeIndex specialization, deduced from function.
 * @param function: Callable to invoke in ascending index order. Return values are discarded.
 */
template <std::uint32_t Count, typename Function>
inline constexpr void for_each_index(Function&& function)
{
    for_each_index(static_cast<Function&&>(function), std::make_integer_sequence<std::uint32_t, Count> {});
}

} // namespace hal::cfg::detail
