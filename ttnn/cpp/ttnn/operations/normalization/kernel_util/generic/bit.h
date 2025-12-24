// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bit.h
 * @brief Useful bit-manipulation utilities
 */

#pragma once

#include <cstring>
#include <type_traits>

namespace norm::kernel_util::generic {
/**
 * @brief C++17 compatible bit_cast replacement using union
 * @tparam To The type to cast to
 * @tparam From The type to cast from
 * @param from The value to cast from
 * @return The casted value
 */
template <
    typename To,
    typename From,
    typename = std::enable_if_t<
        sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<To> && std::is_trivially_copyable_v<From>>>
constexpr To bit_cast(const From& from) {
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}
}  // namespace norm::kernel_util::generic
