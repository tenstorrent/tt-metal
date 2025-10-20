// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bit.h
 * @brief Bit operations for general use.
 */

#pragma once

namespace norm::kernel_util::generic {
/**
 * @brief C++17 compatible bit_cast replacement using union
 * @tparam To The type to cast to
 * @tparam From The type to cast from
 * @param from The value to cast from
 * @return The casted value
 */
template <typename To, typename From>
inline To bit_cast(const From& from) noexcept {
    static_assert(sizeof(To) == sizeof(From), "Types must have same size");
    static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");

    union {
        From f;
        To t;
    } u;

    u.f = from;
    return u.t;
}
}  // namespace norm::kernel_util::generic
