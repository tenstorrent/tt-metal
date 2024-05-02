// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <type_traits>

#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace decorators {

template <template <typename...> class Template, typename T>
struct is_specialization_of : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template, Template<Args...>> : std::true_type {};

std::string to_name(const std::string& fully_qualified_name) {
    std::string last_token = fully_qualified_name.substr(fully_qualified_name.rfind("::") + 2);
    return last_token;
}

std::string to_class_name(const std::string& fully_qualified_name) {
    std::string last_token = fully_qualified_name.substr(fully_qualified_name.rfind("::") + 2);
    return last_token + "_t";
}
template <typename T, typename Return, typename... Args>
constexpr auto resolve_call_method(Return (*launch)(Args...)) {
    return [](const T& self, Args... args) { return self(std::forward<Args>(args)...); };
}

template <typename... py_args_t>
struct pybind_arguments_t {
    std::tuple<py_args_t...> value;

    pybind_arguments_t(py_args_t... args) : value(std::forward_as_tuple(args...)) {}
};

template <typename function_t, typename... py_args_t>
struct pybind_overload_t {
    function_t function;
    pybind_arguments_t<py_args_t...> args;

    pybind_overload_t(function_t function, py_args_t... args) : function{function}, args{args...} {}
};

template <auto id, typename concrete_operation_t, auto... launch_args_t, typename... overload_t>
auto bind_registered_operation(
    py::module& module,
    const operation_t<id, concrete_operation_t, launch_args_t...>& operation,
    const char* doc,
    overload_t&&... overloads) {
    using T = operation_t<id, concrete_operation_t, launch_args_t...>;

    const auto fully_qualified_name = std::string{operation.fully_qualified_name};
    auto name = to_name(fully_qualified_name);              // Get "add" from "ttnn::add"
    auto class_name = to_class_name(fully_qualified_name);  // Convert ttnn::add to ttnn_add

    py::class_<T> py_operation(module, class_name.c_str());
    py_operation.doc() = doc;
    py_operation.def_property_readonly(
        "fully_qualified_name",
        [](const T& self) -> const char* { return self.fully_qualified_name; },
        "Fully qualified name of the api");

    py_operation.def_property_readonly(
        "__ttnn__", [](const T& self) { return std::nullopt; });  // Identifier for the operation

    (
        [&py_operation](auto&& overload) {
            if constexpr (is_specialization_of<pybind_arguments_t, std::decay_t<decltype(overload)>>::value) {
                std::apply(
                    [&py_operation](auto... args) {
                        py_operation.def("__call__", resolve_call_method<T>(&concrete_operation_t::execute), args...);
                    },
                    overload.value);
            } else {
                std::apply(
                    [&py_operation, &overload](auto... args) {
                        py_operation.def("__call__", overload.function, args...);
                    },
                    overload.args.value);
            }
        }(overloads),
        ...);

    module.attr(name.c_str()) = T{operation};  // Bind an instance of the operation to the module

    return py_operation;
}

}  // namespace decorators

using decorators::bind_registered_operation;
using decorators::pybind_arguments_t;
using decorators::pybind_overload_t;

}  // namespace ttnn
