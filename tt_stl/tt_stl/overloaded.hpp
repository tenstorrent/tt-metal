// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace tt::stl {

// `overloaded` allows to combine multiple lambdas into a single object with the overloaded `operator()`.
// This is useful for creating visitor objects for using in `std::visit`, for example:
//
//
// class A, B, C;
// std::variant<A, B, C> my_variant;
// ...
//
// std::visit(
//     tt::stl::overloaded{
//         [](const A&) { /*Process A...*/ },
//         [](const B&) { /*Process B...*/ },
//         [](const C&) { /*Process C...*/ },
//     },
//     my_variant);
//
//

namespace detail {

template <typename T, typename = void>
struct overloaded_base : std::remove_cv<std::remove_reference_t<T>> {};

template <typename T>
struct overloaded_base<T&, std::enable_if_t<not std::is_copy_constructible_v<T>>> {
    using type = std::reference_wrapper<T>;
};

template <typename T>
using overloaded_base_t = typename overloaded_base<T>::type;

}  // namespace detail

template <typename... Ts>
struct overloaded : detail::overloaded_base_t<Ts>... {
    using Ts::operator()...;
};

template <typename... Ts>
overloaded(Ts&&...) -> overloaded<Ts...>;

}  // namespace tt::stl
