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

template <typename T, typename Return, typename... Args>
constexpr auto resolve_call_method(Return (*function)(Args...)) {
    return [](const T& self, Args&&... args) { return self(std::forward<Args>(args)...); };
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

template <typename registered_operation_t, typename concrete_operation_t, typename T, typename... py_args_t>
void define_call_operator(T& py_operation, const pybind_arguments_t<py_args_t...>& overload) {
    std::apply(
        [&py_operation](auto... args) {
            py_operation.def(
                "__call__", resolve_call_method<registered_operation_t>(&concrete_operation_t::operator()), args...);
        },
        overload.value);
}

template <
    typename registered_operation_t,
    typename concrete_operation_t,
    typename T,
    typename function_t,
    typename... py_args_t>
void define_call_operator(T& py_operation, const pybind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) { py_operation.def("__call__", overload.function, args...); },
        overload.args.value);
}

template <
    auto id,
    reflect::fixed_string cpp_fully_qualified_name,
    typename concrete_operation_t,
    bool auto_launch_op,
    typename... overload_t>
auto bind_registered_operation(
    py::module& module,
    const operation_t<id, cpp_fully_qualified_name, concrete_operation_t, auto_launch_op>& operation,
    const std::string& doc,
    overload_t&&... overloads) {
    using registered_operation_t = std::decay_t<decltype(operation)>;
    py::class_<registered_operation_t> py_operation(module, operation.class_name().c_str());

    py_operation.doc() = doc;

    py_operation.def_property_readonly(
        "name",
        [](const registered_operation_t& self) -> const std::string { return self.base_name(); },
        "Shortened name of the api");

    py_operation.def_property_readonly(
        "python_fully_qualified_name",
        [](const registered_operation_t& self) -> const std::string { return self.python_fully_qualified_name(); },
        "Fully qualified name of the api");

    py_operation.def_property_readonly("__ttnn_operation__", [](const registered_operation_t& self) {
        return std::nullopt;
    });  // Attribute to identify of ttnn operations

    (
        [&py_operation](auto&& overload) {
            define_call_operator<registered_operation_t, concrete_operation_t>(py_operation, overload);
        }(overloads),
        ...);

    module.attr(operation.base_name().c_str()) = operation;  // Bind an instance of the operation to the module

    return py_operation;
}

}  // namespace decorators

using decorators::bind_registered_operation;
using decorators::pybind_arguments_t;
using decorators::pybind_overload_t;

}  // namespace ttnn
