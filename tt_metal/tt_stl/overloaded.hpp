// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace tt::stl {

// `overloaded` allows to combine multiple lambdas into a single object with the overloaded `operator()`.
// This is useful for creating visitor objects for using in `std::visit`, for example:
//
// void f(const std::variant<int, std::string>& v) {
//   std::visit(overloaded {
//     [](int i) { std::cout << "Got an int: " << i << std::endl; },
//     [](const std::string& str) { std::cout << "Got a string: " << str << std::endl; },
//   }, v);
// }
//
//
// Generic lambdas can also be used to simplify handling of variant types:
//
// void f(const std::variant<std::vector<int>, int, std::string>& v) {
//   std::visit(overloaded {
//     [](const std::vector<int>& vec) { std::cout << "Got a vector of size: " << vec.size() << std::endl; },
//     [](const auto& other) { std::cout << "Got something else: " << other << std::endl; },
//   }, v);
// }
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
