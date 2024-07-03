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
constexpr auto resolve_call_method(Return (*execute_on_worker_thread)(Args...)) {
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
void add_operator_call(T& py_operation, const pybind_arguments_t<py_args_t...>& overload) {
    std::apply(
        [&py_operation](auto... args) {
            py_operation.def(
                "__call__",
                resolve_call_method<registered_operation_t>(&concrete_operation_t::execute_on_worker_thread),
                args...);
        },
        overload.value);
}

template <
    typename registered_operation_t,
    typename concrete_operation_t,
    typename T,
    typename function_t,
    typename... py_args_t>
void add_operator_call(T& py_operation, const pybind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) { py_operation.def("__call__", overload.function, args...); },
        overload.args.value);
}

template <auto id, typename concrete_operation_t>
std::string append_input_tensor_schemas_to_doc(
    const operation_t<id, concrete_operation_t>& operation, const std::string& doc) {
    std::stringstream updated_doc;

    auto write_row = [&updated_doc]<typename Tuple>(const Tuple& tuple) {
        auto index = 0;

        std::apply(
            [&index, &updated_doc](const auto&... args) {
                (
                    [&index, &updated_doc](const auto& item) {
                        updated_doc << "        ";
                        if (index == 0) {
                            updated_doc << " * - ";
                        } else {
                            updated_doc << "   - ";
                        }
                        updated_doc << fmt::format("{}", item);
                        updated_doc << "\n";
                        index++;
                    }(args),
                    ...);
            },
            tuple);
    };

    if constexpr (detail::has_input_tensor_schemas<concrete_operation_t>()) {
        if constexpr (std::tuple_size_v<decltype(concrete_operation_t::input_tensor_schemas())> > 0) {
            updated_doc << doc << "\n\n";
            auto tensor_index = 0;
            for (const auto& schema : concrete_operation_t::input_tensor_schemas()) {
                updated_doc << "    .. list-table:: Input Tensor " << tensor_index << "\n\n";
                write_row(ttnn::TensorSchema::attribute_names());
                write_row(schema.attribute_values());
                tensor_index++;
                updated_doc << "\n";
            }
            updated_doc << "\n";
            return updated_doc.str();
        } else {
            return doc;
        }
    } else {
        return doc;
    }
}

auto bind_registered_operation_helper(
    py::module& module, const auto& operation, const std::string& doc, auto attach_call_operator) {
    using registered_operation_t = std::decay_t<decltype(operation)>;

    const auto cpp_fully_qualified_name = std::string{operation.cpp_fully_qualified_name};

    py::class_<registered_operation_t> py_operation(module, operation.class_name().c_str());

    if constexpr (requires { append_input_tensor_schemas_to_doc(operation, doc); }) {
        py_operation.doc() = append_input_tensor_schemas_to_doc(operation, doc).c_str();
    } else {
        py_operation.doc() = doc;
    }

    py_operation.def_property_readonly(
        "name",
        [](const registered_operation_t& self) -> const std::string { return self.name(); },
        "Shortened name of the api");

    py_operation.def_property_readonly(
        "python_fully_qualified_name",
        [](const registered_operation_t& self) -> const std::string { return self.python_fully_qualified_name(); },
        "Fully qualified name of the api");

    py_operation.def_property_readonly("__ttnn_operation__", [](const registered_operation_t& self) {
        return std::nullopt;
    });  // Attribute to identify of ttnn operations

    attach_call_operator(py_operation);

    module.attr(operation.name().c_str()) = operation;  // Bind an instance of the operation to the module

    return py_operation;
}

template <auto id, typename concrete_operation_t, typename... overload_t>
auto bind_registered_operation(
    py::module& module,
    const operation_t<id, concrete_operation_t>& operation,
    const std::string& doc,
    overload_t&&... overloads) {
    using registered_operation_t = operation_t<id, concrete_operation_t>;

    auto attach_call_operator = [&](auto& py_operation) {
        (
            [&py_operation](auto&& overload) {
                add_operator_call<registered_operation_t, concrete_operation_t>(py_operation, overload);
            }(overloads),
            ...);
    };

    return bind_registered_operation_helper(module, operation, doc, attach_call_operator);
}

template <auto id, typename lambda_t, typename... overload_t>
auto bind_registered_operation(
    py::module& module,
    const lambda_operation_t<id, lambda_t>& operation,
    const std::string& doc,
    overload_t&&... overloads) {
    using registered_operation_t = lambda_operation_t<id, lambda_t>;

    auto attach_call_operator = [&](auto& py_operation) {
        (
            [&py_operation](auto&& overload) {
                std::apply(
                    [&py_operation, &overload](auto&&... args) {
                        py_operation.def("__call__", overload.function, args...);
                    },
                    overload.args.value);
            }(overloads),
            ...);
    };

    return bind_registered_operation_helper(module, operation, doc, attach_call_operator);
}

}  // namespace decorators

using decorators::bind_registered_operation;
using decorators::pybind_arguments_t;
using decorators::pybind_overload_t;

}  // namespace ttnn
