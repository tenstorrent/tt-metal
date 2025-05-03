// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>
namespace tt::tt_metal {

template <typename T>
bool is_power_of_two(T val) {
    return (val & (val - 1)) == T(0);
}

template <typename T>
bool is_power_of_two_at_least(T val, T power2) {
    TT_ASSERT(is_power_of_two(power2));
    return (val & (val - power2)) == T(0);
}

template <typename T>
bool is_power_of_two_at_least_32(T val) {
    return is_power_of_two_at_least(val, T(32));
}

}  // namespace tt::tt_metal
