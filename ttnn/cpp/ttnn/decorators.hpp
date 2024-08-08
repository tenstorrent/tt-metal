// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <reflect>

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "ttnn/core.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace decorators {

namespace detail {


template <typename... args_t>
void log(const std::string& prefix, args_t&&... args) {
    auto args_tuple = std::tuple{[](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_pointer_v<T>) {
            return fmt::format("{}", tt::stl::get_type_name<T>());
        } else {
            return arg;
        }
    }(std::forward<args_t>(args))...};

    std::string fmt{prefix};
    fmt += "\n";
    for (int i = 0; i < sizeof...(args); i++) {
        fmt += fmt::format("\t{:2}: {}\n", i, "{}");
    }
    std::apply([&fmt](const auto&... args) { tt::log_debug(tt::LogOp, fmt.c_str(), args...); }, args_tuple);
}

// Get "add" from "ttnn::add"
static const std::string base_name(const std::string& cpp_fully_qualified_name) {
    auto last_token = cpp_fully_qualified_name.substr(cpp_fully_qualified_name.rfind("::") + 2);
    return last_token;
}

// Convert "ttnn::add" to "add_t"
static const std::string class_name(const std::string& cpp_fully_qualified_name) {
    return base_name(cpp_fully_qualified_name) + "_t";
}

// Convert "ttnn::add" to "ttnn.add"
static const std::string python_fully_qualified_name(const std::string& cpp_fully_qualified_name) {
    auto replace = [](const std::string& input, const std::string& from, const std::string& to) {
        if (from.empty()) {
            return input;
        }
        auto output = input;
        size_t start = 0;
        while ((start = output.find(from, start)) != std::string::npos) {
            output.replace(start, from.length(), to);
            start += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
        };
        return output;
    };
    return replace(cpp_fully_qualified_name, "::", ".");
}

}  // namespace detail

template <auto id, reflect::fixed_string cpp_fully_qualified_name, typename concrete_operation_t, bool auto_launch_op>
struct operation_t {
    template <typename... args_t>
    auto operator()(args_t&&... args) const {
        ZoneScopedN("Run ttnn operation ");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data.data()), cpp_fully_qualified_name.size());
        tt::log_debug(tt::LogOp, "Started   C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});

        // #8479: Fix and re-enable logging in cpp operation decorator
        // detail::log("Arguments: ", std::forward<args_t>(args)...);

        auto output = concrete_operation_t::operator()(std::forward<decltype(args)>(args)...);
        tt::log_debug(tt::LogOp, "Finished  C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        return output;
    }

    // Get "add" from "ttnn::add"
    const std::string base_name() const { return detail::base_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "add_t"
    const std::string class_name() const { return detail::class_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "ttnn.add"
    const std::string python_fully_qualified_name() const {
        return detail::python_fully_qualified_name(std::string{cpp_fully_qualified_name});
    }
};

template <reflect::fixed_string cpp_fully_qualified_name, typename concrete_operation_t>
constexpr auto register_operation() {
    return operation_t<__COUNTER__, cpp_fully_qualified_name, concrete_operation_t, false>{};
}

template <reflect::fixed_string cpp_fully_qualified_name, typename concrete_operation_t>
constexpr auto register_operation_with_auto_launch_op() {
    return operation_t<__COUNTER__, cpp_fully_qualified_name, concrete_operation_t, true>{};
}

namespace detail {
template <auto lambda_t>
struct lambda_operation_t {
    static auto operator()(auto&&... args) { return lambda_t(std::forward<decltype(args)>(args)...); }
};
}  // namespace detail

// If you are feeling lazy, you can use this macro to create an operation struct from a lambda
// You  will have to implement async manually
#define REGISTER_OPERATION_FROM_FUNCTION(cpp_fully_qualified_name, function) \
    (::ttnn::decorators::register_operation<                                 \
        cpp_fully_qualified_name,                                            \
        ::ttnn::decorators::detail::lambda_operation_t<[](auto&&... args) {  \
            return function(std::forward<decltype(args)>(args)...);          \
        }>>())

}  // namespace decorators

using ttnn::decorators::register_operation;
using ttnn::decorators::register_operation_with_auto_launch_op;

}  // namespace ttnn
