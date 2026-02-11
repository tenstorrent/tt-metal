// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace ttnn {

namespace nb = nanobind;

// Holds a function pointer and its nanobind argument specs
template <typename Func, typename... Args>
struct overload_t {
    Func func;
    std::tuple<Args...> args;

    constexpr overload_t(Func f, const Args&... a) : func(f), args(std::make_tuple(a...)) {}
};

// Deduction guide
template <typename Func, typename... Args>
overload_t(Func, Args...) -> overload_t<Func, Args...>;

namespace detail {

// Wrapper class that holds operation metadata and is bound as a callable Python object
struct function_wrapper_t {
    const char* name_;
    const char* py_name_;
};

// Helper to wrap a free function as a method (adding ignored self parameter)
template <typename Wrapper, typename Ret, typename... FuncArgs>
auto make_method_wrapper(Ret (*func)(FuncArgs...)) {
    return [func](const Wrapper& /*self*/, FuncArgs... args) -> Ret { return func(std::forward<FuncArgs>(args)...); };
}

// Add a single __call__ overload to the class
template <typename Wrapper, typename Class, typename Func, typename... Args>
void add_call_overload(Class& cls, const overload_t<Func, Args...>& spec) {
    auto method = make_method_wrapper<Wrapper>(spec.func);
    std::apply([&cls, &method](const Args&... args) { cls.def("__call__", method, args...); }, spec.args);
}

}  // namespace detail

// C++20 non-type template parameter for string literals
// This ensures each operation name creates a unique type across all translation units.
// Uniqueness is determined by BOTH the string length (N) AND the actual character content.
// For example: "split" (N=6) and "matmul" (N=7) are different types, and
//              "split" and "split" are the same type (same N and same characters).
//              "test" and "tset" are different types (same N by different characters).
template <std::size_t N>
struct unique_string {
    std::array<char, N> data;

    constexpr unique_string(const char (&str)[N]) { std::copy_n(str, N, data.begin()); }

    constexpr operator const char*() const { return data.data(); }
};

// Helper struct template - each unique operation name creates a unique type
template <unique_string Name>
struct unique_wrapper_base {
    std::string name_;
    std::string py_name_;
};

// Main binding function - binds a set of C++ function overloads as a callable Python object
//
// Usage:
//   ttnn::bind_function<"split">(mod, doc, ttnn::overload_t(...))
//   ttnn::bind_function<"some_op", "ttnn.experimental">(mod, doc, ttnn::overload_t(...))
//
// The FuncName template parameter uses C++20 unique_string to ensure each operation
// gets a unique type across all translation units, preventing mangled name collisions.
// The fully qualified Python name is automatically constructed as Namespace + FuncName.
// Default namespace is "ttnn."; use "ttnn.experimental." for experimental operations.
template <unique_string FuncName, unique_string Namespace = unique_string{"ttnn."}, typename... Overloads>
void bind_function(nb::module_& mod, const char* doc, Overloads&&... overloads) {
    // Create a unique wrapper type using the operation name
    // Each operation name creates a distinct type, ensuring uniqueness across TUs
    using wrapper_t = unique_wrapper_base<FuncName>;

    std::string class_name = std::string(FuncName) + "_t";
    std::string python_fully_qualified_name = std::string(Namespace) + std::string(FuncName);

    auto cls = nb::class_<wrapper_t>(mod, class_name.c_str());
    cls.def_prop_ro("name", [](const wrapper_t& self) { return self.name_; });
    cls.def_prop_ro("python_fully_qualified_name", [](const wrapper_t& self) { return self.py_name_; });

    // Marker attribute for Python-side auto-registration
    cls.def_prop_ro("__ttnn_operation__", [](const wrapper_t&) { return nb::none(); });

    cls.doc() = doc;

    // Add __call__ for each overload
    (detail::add_call_overload<wrapper_t>(cls, std::forward<Overloads>(overloads)), ...);

    // Create instance and bind to module
    // Each unique type (per operation name) has its own static instance
    static wrapper_t instance{std::string(FuncName), python_fully_qualified_name};
    mod.attr(FuncName) = &instance;
}

}  // namespace ttnn
