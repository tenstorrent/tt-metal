// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

// Main binding function - binds a set of C++ function overloads as a callable Python object
//
// Usage:
//   ttnn::bind_function(
//       mod,
//       "split",
//       "ttnn.split",
//       "Documentation...",
//       ttnn::overload_t(
//           nb::overload_cast<const Tensor&, int64_t, int64_t, const std::optional<MemoryConfig>&>(&ttnn::split),
//           nb::arg("input_tensor"), nb::arg("split_size"), ...),
//       ttnn::overload_t(
//           nb::overload_cast<const Tensor&, const SmallVector<int64_t>&, int64_t, const
//           std::optional<MemoryConfig>&>(&ttnn::split), nb::arg("input_tensor"), nb::arg("split_sizes"), ...)
//   );
//
template <typename... Overloads>
void bind_function(
    nb::module_& mod,
    const char* name,
    const char* python_fully_qualified_name,
    const char* doc,
    Overloads&&... overloads) {
    // Create a unique wrapper type for this function (using name length as discriminator)
    // In practice, each bind_function call creates its own static instance
    struct wrapper_t {
        const char* name_;
        const char* py_name_;
    };

    // Generate class name: "split" -> "split_t"
    std::string class_name = std::string(name) + "_t";

    auto cls = nb::class_<wrapper_t>(mod, class_name.c_str());

    // Add standard properties that match registered_operation_t interface
    cls.def_prop_ro("name", [](const wrapper_t& self) { return std::string(self.name_); });
    cls.def_prop_ro("python_fully_qualified_name", [](const wrapper_t& self) { return std::string(self.py_name_); });

    // Marker attribute for Python-side auto-registration
    cls.def_prop_ro("__ttnn_operation__", [](const wrapper_t&) { return nb::none(); });

    // Set docstring
    cls.doc() = doc;

    // Add __call__ for each overload
    (detail::add_call_overload<wrapper_t>(cls, std::forward<Overloads>(overloads)), ...);

    // Create static instance and bind to module
    static wrapper_t instance{name, python_fully_qualified_name};
    mod.attr(name) = &instance;
}

}  // namespace ttnn
