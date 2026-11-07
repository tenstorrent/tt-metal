// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace hal
{

// TODO issue #58443: Remove the duplicate ckernel::to_underlying after HAL migration.
/** @brief Convert an enum value to its underlying integral type. */
template <typename T>
constexpr auto to_underlying(T value) noexcept
{
    return static_cast<std::underlying_type_t<T>>(value);
}

} // namespace hal
