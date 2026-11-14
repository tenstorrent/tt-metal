// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace hal
{

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
