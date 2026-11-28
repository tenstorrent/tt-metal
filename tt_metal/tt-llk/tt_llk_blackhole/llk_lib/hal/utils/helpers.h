// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

namespace hal
{

namespace detail
{
// TODO: #58443 Remove the duplicate ckernel::is_valid.
/** @brief Return whether a field value fits in its encoding width without truncation. */
template <typename T>
constexpr bool is_valid(const T val, const std::uint8_t wid)
{
    const T mask = (1 << wid) - 1;
    return (val & mask) == val;
}
} // namespace detail

// TODO: #58443 Remove the duplicate ckernel::to_underlying.
/** @brief Convert an enum value to its underlying integral type. */
template <typename T>
constexpr auto to_underlying(T value) noexcept
{
    return static_cast<std::underlying_type_t<T>>(value);
}

/** @brief Dependent false value for rejecting a template instantiation with static_assert. */
template <auto>
inline constexpr bool always_false = false;

} // namespace hal
