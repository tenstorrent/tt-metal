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
#include "ttnn/common/constants.hpp"
#include "ttnn/device_operation.hpp"

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

// Primitive operations map directly to device operations
template <typename operation_t>
concept PrimitiveOperationConcept = device_operation::DeviceOperationConcept<operation_t>;


// Composite operation allows any code to be executed
template<typename operation_t>
concept CompositeOperationConcept = !PrimitiveOperationConcept<operation_t>;

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
struct registered_operation_t {
    static constexpr auto is_primitive = PrimitiveOperationConcept<operation_t>;

    // Get "add" from "ttnn::add"
    const std::string base_name() const { return detail::base_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "add_t"
    const std::string class_name() const { return detail::class_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "ttnn.add"
    const std::string python_fully_qualified_name() const {
        return detail::python_fully_qualified_name(std::string{cpp_fully_qualified_name});
    }

    template <typename... args_t>
    requires PrimitiveOperationConcept<operation_t>
    auto invoke(uint8_t queue_id, args_t&&... args) const {
        static_assert(requires { operation_t::invoke(std::forward<decltype(args)>(args)...); },
                      "Primitive Operation must implement operator() method to be invoked.");
        ZoneScopedN("Run primitive ttnn operation");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data.data()), cpp_fully_qualified_name.size());
        auto [operation_attributes, tensors_args] = operation_t::invoke(std::forward<decltype(args)>(args)...);
        return ttnn::device_operation::run<operation_t>(queue_id, operation_attributes, tensors_args);
    }

    template <typename... args_t>
    requires(PrimitiveOperationConcept<operation_t>)
    auto invoke(args_t&&... args) const {
        return invoke(DefaultQueueId, std::forward<args_t>(args)...);
    }

    template <typename... args_t>
    requires(CompositeOperationConcept<operation_t>)
    auto invoke(args_t&&... args) const {
        ZoneScopedN("Run composite ttnn operation ");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data.data()), cpp_fully_qualified_name.size());
        return operation_t::invoke(std::forward<decltype(args)>(args)...);
    }

    template <typename... args_t>
    auto operator()(args_t&&... args) const {
        tt::log_debug(tt::LogOp, "Started   C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        auto output = invoke(std::forward<args_t>(args)...);
        tt::log_debug(tt::LogOp, "Finished  C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        return output;
    }

};

template<reflect::fixed_string cpp_fully_qualified_name>
struct operation_name_key_t{
    friend consteval auto get(operation_name_key_t<cpp_fully_qualified_name>);
};

template<typename operation_t>
struct operation_key_t{
    friend consteval auto get(operation_key_t<operation_t>);
};

template<reflect::fixed_string cpp_fully_qualified_name, typename operation_t, auto operation>
struct set_operation_t : std::true_type {
    friend consteval auto get(operation_key_t<operation_t>) {
        return operation;
    }
    friend consteval auto get(operation_name_key_t<cpp_fully_qualified_name>) {
        return operation;
    }
};

constexpr reflect::fixed_string prim_namespace = "ttnn::prim";

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
consteval void assert_operation_in_correct_namespace() {
    if constexpr (PrimitiveOperationConcept<operation_t>) {
        if constexpr(cpp_fully_qualified_name.size() > sizeof(prim_namespace)) {
            constexpr auto namespace_substring = tt::stl::reflection::fixed_string_substring<0, sizeof(prim_namespace)>(cpp_fully_qualified_name);
            static_assert(tt::stl::reflection::fixed_string_equals(namespace_substring, prim_namespace), "Primitive operations must be in the `ttnn::prim` namespace.");
        } else {
            #ifndef DISABLE_NAMESPACE_STATIC_ASSERT
            static_assert(false, "Primitive operations must be in the `ttnn::prim` namespace.");
            #endif
        }
    } else {
        if constexpr (cpp_fully_qualified_name.size() > sizeof(prim_namespace)) {
            constexpr auto namespace_substring = tt::stl::reflection::fixed_string_substring<0, sizeof(prim_namespace)>(cpp_fully_qualified_name);
            static_assert(not tt::stl::reflection::fixed_string_equals(namespace_substring, prim_namespace), "Composite operations must not be in the `ttnn::prim` namespace.");
        }
    }
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation_impl() {
    assert_operation_in_correct_namespace<cpp_fully_qualified_name, operation_t>();
    constexpr auto operation = registered_operation_t<cpp_fully_qualified_name, operation_t>{};
    static_assert(not requires(operation_name_key_t<cpp_fully_qualified_name> key) { get(key); }, "Operation with this `cpp_fully_qualified_name` was already registered. Please use a different name.");
    static_assert(not requires(operation_key_t<operation_t> key) { get(key); }, "Operation with this `operation_t` was already registered. Please use a different type.");
    static_assert(set_operation_t<cpp_fully_qualified_name, operation_t, operation>::value);
    return operation;
}


template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation() {
    return register_operation_impl<cpp_fully_qualified_name, operation_t>();
}

namespace detail {
template <auto lambda_t>
struct lambda_operation_t {
    static auto invoke(auto&&... args) { return lambda_t(std::forward<decltype(args)>(args)...); }
};
}  // namespace detail

}  // namespace decorators

using ttnn::decorators::register_operation;
using ttnn::decorators::register_operation;

}  // namespace ttnn
