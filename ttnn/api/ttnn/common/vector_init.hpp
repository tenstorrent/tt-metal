// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <type_traits>

template <typename T, typename... Vs>
std::vector<T> vector_init(Vs&&... init_values)
    requires requires { std::disjunction_v<std::is_same<T, Vs>...>; }
{
    std::vector<T> vec;
    vec.reserve(sizeof...(Vs));
    (vec.emplace_back(std::forward<Vs>(init_values)), ...);
    return vec;
}
