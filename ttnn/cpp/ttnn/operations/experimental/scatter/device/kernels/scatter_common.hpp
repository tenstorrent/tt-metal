// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

constexpr uint32_t ONE_TILE = 1;
constexpr uint32_t FIRST_TILE = 0;

template <typename T>
FORCE_INLINE T read(volatile tt_l1_ptr T* l1_ptr, std::size_t n) {
    return l1_ptr[n];
}

template <typename T>
FORCE_INLINE write(volatile tt_l1_ptr T* l1_ptr, std::size_t n, T value) {
    l1_ptr[n] = value;
}
