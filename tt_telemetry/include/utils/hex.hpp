// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <type_traits>

/**
 * @brief Convert an integer value to a hexadecimal string representation
 *
 * This templated function converts any integer type to a hexadecimal string
 * with "0x" prefix. The number of digits is automatically determined based
 * on the size of the integer type (2 digits per byte).
 *
 * @tparam T Integer type (signed or unsigned)
 * @param value The integer value to convert
 * @return std::string Hexadecimal representation with "0x" prefix
 *
 * Examples:
 *   hex(uint8_t(255))    -> "0xff"
 *   hex(uint16_t(4095))  -> "0x0fff"
 *   hex(uint32_t(255))   -> "0x000000ff"
 *   hex(int32_t(-1))     -> "0xffffffff"
 *   hex(uint64_t(255))   -> "0x00000000000000ff"
 */
template <typename T>
std::string hex(T value) {
    static_assert(std::is_integral_v<T>, "hex() can only be used with integer types");

    std::ostringstream oss;
    oss << "0x" << std::hex << std::setfill('0') << std::setw(sizeof(T) * 2)  // 2 hex digits per byte
        << static_cast<typename std::make_unsigned<T>::type>(value);

    return oss.str();
}
