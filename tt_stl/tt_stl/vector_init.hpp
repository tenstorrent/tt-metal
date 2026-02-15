// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <type_traits>

#include <tt_stl/strong_type.hpp>

namespace ttsl {

using vector_size = StrongType<size_t, struct VectorSizeTag>;

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

template <typename T, typename... Vs>
std::vector<T> vector_init(const vector_size reserve_count, Vs&&... init_values) {
    std::vector<T> vec;
    vec.reserve(*reserve_count);
    (vec.emplace_back(std::forward<Vs>(init_values)), ...);
    return vec;
}

}  // namespace ttsl
