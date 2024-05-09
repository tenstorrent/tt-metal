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
}

template <auto id, typename concrete_operation_t, typename... overload_t>
auto bind_registered_operation(
    py::module& module,
    const operation_t<id, concrete_operation_t>& operation,
    const std::string& doc,
    overload_t&&... overloads) {
    using registered_operation_t = operation_t<id, concrete_operation_t>;

    const auto cpp_fully_qualified_name = std::string{operation.cpp_fully_qualified_name};

    py::class_<registered_operation_t> py_operation(module, operation.class_name().c_str());

    py_operation.doc() = append_input_tensor_schemas_to_doc(operation, doc).c_str();

    py_operation.def_property_readonly(
        "name",
        [](const registered_operation_t& self) -> const std::string { return self.name(); },
        "Shortened name of the api");

    py_operation.def_property_readonly(
        "python_fully_qualified_name",
        [](const registered_operation_t& self) -> const std::string { return self.python_fully_qualified_name(); },
        "Fully qualified name of the api");

    py_operation.def_property_readonly(
        "__ttnn__", [](const registered_operation_t& self) { return std::nullopt; });  // Identifier for the operation

    (
        [&py_operation](auto&& overload) {
            if constexpr (is_specialization_of<pybind_arguments_t, std::decay_t<decltype(overload)>>::value) {
                std::apply(
                    [&py_operation](auto... args) {
                        py_operation.def("__call__", resolve_call_method<registered_operation_t>(&concrete_operation_t::execute), args...);
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

    module.attr(operation.name().c_str()) = operation;  // Bind an instance of the operation to the module

    return py_operation;
}

}  // namespace decorators

using decorators::bind_registered_operation;
using decorators::pybind_arguments_t;
using decorators::pybind_overload_t;

}  // namespace ttnn
