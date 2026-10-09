// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace hal
{

/**
 * @brief Convert an enum value to its underlying integral type.
 * @todo Remove the duplicate ckernel::to_underlying after HAL migration:
 *       https://github.com/tenstorrent/tt-metal/issues/58443
 */
template <typename T>
constexpr auto to_underlying(T value) noexcept
{
    return static_cast<std::underlying_type_t<T>>(value);
}

} // namespace hal
