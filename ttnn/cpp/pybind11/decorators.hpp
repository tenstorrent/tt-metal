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

template <typename T, typename return_t, typename... args_t>
constexpr auto resolve_call_method(return_t (*function)(args_t...)) {
    return [](const T& self, args_t... args) {
        return self(std::forward<args_t>(args)...);
    };
}

template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template <typename... TArgs>
struct arg_traits {};

template <typename ClassType, typename ReturnType, typename... args_t>
struct function_traits<ReturnType (ClassType::*)(args_t...) const>
// we specialize for pointers to member function
{
    using return_t = ReturnType;
    using arg_tuple = arg_traits<args_t...>;
};

template <typename F>
constexpr auto resolve_primitive_operation_call_method(F) {
    using traits = function_traits<F>;

    return []<typename TSelf, typename... TArgs>(arg_traits<TSelf, TArgs...>) {
        return [](TSelf self, TArgs... args, std::uint8_t queue_id) -> typename traits::return_t {
            return self(queue_id, static_cast<decltype(args)&&>(args)...);
        };
    }(typename traits::arg_tuple{});
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

template <typename registered_operation_t, typename operation_t, typename py_operation_t, typename... py_args_t>
void def_call_operator(py_operation_t& py_operation, const pybind_arguments_t<py_args_t...>& overload) {
    std::apply(
        [&py_operation](auto... args) {
            py_operation.def("__call__", resolve_call_method<registered_operation_t>(&operation_t::invoke), args...);
        },
        overload.value);
}

template <typename registered_operation_t,
          typename operation_t,
          typename py_operation_t,
          typename function_t,
          typename... py_args_t>
    requires PrimitiveOperationConcept<operation_t>
void def_call_operator(py_operation_t& py_operation, const pybind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) {
            py_operation.def("__call__",
                             resolve_primitive_operation_call_method(overload.function),
                             args...,
                             py::arg("queue_id") = 0);
        },
        overload.args.value);
}

template <typename registered_operation_t,
          typename operation_t,
          typename py_operation_t,
          typename function_t,
          typename... py_args_t>
    requires CompositeOperationConcept<operation_t>
void def_call_operator(py_operation_t& py_operation, const pybind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) {
            py_operation.def("__call__", overload.function, args...);
        },
        overload.args.value);
}

template <typename py_operation_t, typename function_t, typename... py_args_t>
void def_primitive_operation_method(py_operation_t& py_operation,
                                    const pybind_overload_t<function_t, py_args_t...>& overload,
                                    auto name,
                                    auto method) {
    std::apply(
        [&py_operation, &overload, &name, &method](auto... args) {
            py_operation.def(name, resolve_primitive_operation_method(overload.function, method), args...);
        },
        overload.args.value);
}

template <reflect::fixed_string cpp_fully_qualified_name,
          typename operation_t,
          bool auto_launch_op,
          typename... overload_t>
auto bind_registered_operation(
    py::module& module,
    const registered_operation_t<cpp_fully_qualified_name, operation_t, auto_launch_op>& operation,
    const std::string& doc,
    overload_t&&... overloads) {
    using registered_operation_t = std::decay_t<decltype(operation)>;
    py::class_<registered_operation_t> py_operation(module, operation.class_name().c_str());

    py_operation.doc() = doc;

    py_operation.def_property_readonly(
        "name",
        [](const registered_operation_t& self) -> const std::string {
            return self.base_name();
        },
        "Shortened name of the api");

    py_operation.def_property_readonly(
        "python_fully_qualified_name",
        [](const registered_operation_t& self) -> const std::string {
            return self.python_fully_qualified_name();
        },
        "Fully qualified name of the api");

    // Attribute to identify of ttnn operations
    py_operation.def_property_readonly("__ttnn_operation__", [](const registered_operation_t& self) {
        return std::nullopt;
    });

    py_operation.def_property_readonly(
        "is_primitive",
        [](const registered_operation_t& self) -> bool {
            return registered_operation_t::is_primitive;
        },
        "Specifies if the operation maps to a single program");

    (
        [&py_operation](auto&& overload) {
            def_call_operator<registered_operation_t, operation_t>(py_operation, overload);
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
