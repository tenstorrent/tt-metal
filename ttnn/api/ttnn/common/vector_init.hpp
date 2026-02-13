// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <type_traits>

template <typename F, typename V, typename... Vs>
struct expand_call {
    static void call(F& f, V&& v, Vs&&... vs) {
        f(v);
        expand_call<F, Vs...>::call(f, std::forward<Vs>(vs)...);
    }
};

template <typename F, typename V>
struct expand_call<F, V> {
    static void call(F& f, V&& v) { f(std::forward<V>(v)); }
};

template <typename T, typename... Vs>
std::vector<T> vector_init(Vs&&... init_values)
    requires requires { std::disjunction_v<std::is_same<T, Vs>...>; }
{
    std::vector<T> vec;
    vec.reserve(sizeof...(Vs));
    auto emplace = [&vec](auto&& v) { vec.emplace_back(std::forward<decltype(v)>(v)); };
    expand_call<decltype(emplace), Vs...>::call(emplace, std::forward<Vs>(init_values)...);
    return vec;
}
