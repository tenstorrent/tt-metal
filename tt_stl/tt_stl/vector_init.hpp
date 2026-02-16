// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <type_traits>

#include <tt_stl/small_vector.hpp>
#include <tt_stl/strong_type.hpp>

namespace ttsl {

using vector_size = StrongType<size_t, struct VectorSizeTag>;

// Generic container_init: works with any container that has reserve() and emplace_back()
// (e.g. std::vector, ttsl::SmallVector)

template <typename Container, size_t N, typename... Vs>
Container container_init(Vs&&... init_values)
    requires requires {
        std::disjunction_v<std::is_same<typename Container::value_type, Vs>...>;
        requires N >= sizeof...(Vs);
        requires sizeof...(Vs) > 0;
    }
{
    Container vec;
    vec.reserve(N);
    (vec.emplace_back(std::forward<Vs>(init_values)), ...);
    return vec;
}

template <typename Container, typename... Vs>
Container container_init(Vs&&... init_values) {
    return container_init<Container, sizeof...(Vs), Vs...>(std::forward<Vs>(init_values)...);
}

template <typename Container, typename... Vs>
Container container_init(const vector_size reserve_count, Vs&&... init_values)
    requires requires {
        std::disjunction_v<std::is_same<typename Container::value_type, Vs>...>;
        requires sizeof...(Vs) > 0;
    }
{
    Container vec;
    vec.reserve(*reserve_count);
    (vec.emplace_back(std::forward<Vs>(init_values)), ...);
    return vec;
}

template <typename T, size_t N, typename... Vs>
std::vector<T> vector_init(Vs&&... init_values) {
    return container_init<std::vector<T>, N>(std::forward<Vs>(init_values)...);
}

template <typename T, typename... Vs>
std::vector<T> vector_init(Vs&&... init_values) {
    return container_init<std::vector<T>>(std::forward<Vs>(init_values)...);
}

template <typename T, typename... Vs>
std::vector<T> vector_init(const vector_size reserve_count, Vs&&... init_values) {
    return container_init<std::vector<T>>(reserve_count, std::forward<Vs>(init_values)...);
}

template <typename T, size_t N, typename... Vs>
SmallVector<T> small_vector_init(Vs&&... init_values) {
    return container_init<SmallVector<T>, N>(std::forward<Vs>(init_values)...);
}

template <typename T, typename... Vs>
SmallVector<T> small_vector_init(Vs&&... init_values) {
    return container_init<SmallVector<T>>(std::forward<Vs>(init_values)...);
}

template <typename T, typename... Vs>
SmallVector<T> small_vector_init(const vector_size reserve_count, Vs&&... init_values) {
    return container_init<SmallVector<T>>(reserve_count, std::forward<Vs>(init_values)...);
}

}  // namespace ttsl
