// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
//
template <typename... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace tt::stl
