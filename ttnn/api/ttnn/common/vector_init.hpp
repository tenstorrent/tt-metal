// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <type_traits>

template <typename T, size_t N, typename... Vs>
std::vector<T> vector_init(Vs&&... init_values)
    requires requires {
        std::disjunction_v<std::is_same<T, Vs>...>;
        requires N >= sizeof...(Vs);
    }
{
    std::vector<T> vec;
    vec.reserve(N);
    (vec.emplace_back(std::forward<Vs>(init_values)), ...);
    return vec;
}

template <typename T, typename... Vs>
std::vector<T> vector_init(Vs&&... init_values) {
    return vector_init<T, sizeof...(Vs), Vs...>(std::forward<Vs>(init_values)...);
}
