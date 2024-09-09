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
    return [](const T& self, args_t... args) { return self(std::forward<args_t>(args)...); };
}


template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())>
{};

template <typename... TArgs>
struct arg_traits {};

template <typename ClassType, typename ReturnType, typename... args_t>
struct function_traits<ReturnType(ClassType::*)(args_t...) const>
// we specialize for pointers to member function
{
    using return_t = ReturnType;
    using arg_tuple = arg_traits<args_t...>;
};

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

template <
    typename registered_operation_t,
    typename operation_t,
    typename py_operation_t,
    typename function_t,
    typename... py_args_t>
void def_call_operator(py_operation_t& py_operation, const pybind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) { py_operation.def("__call__", overload.function, args...); },
        overload.args.value);
}

template <
    reflect::fixed_string cpp_fully_qualified_name,
    typename operation_t,
    bool auto_launch_op,
    typename... overload_t>
auto bind_registered_operation_impl(
    py::module& module,
    const registered_operation_t<cpp_fully_qualified_name, operation_t, auto_launch_op>& operation,
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

template <
    reflect::fixed_string cpp_fully_qualified_name,
    typename operation_t,
    bool auto_launch_op,
    typename... overload_t>
auto bind_registered_operation(
    py::module& module,
    const registered_operation_t<cpp_fully_qualified_name, operation_t, auto_launch_op>& operation,
    const std::string& doc,
    overload_t&&... overloads) {
    static_assert(
        CompositeOperationConcept<operation_t>,
        "Operation must be a composite operation. Use bind_primitive_registered_operation for primitive operations");
    return bind_registered_operation_impl(module, operation, doc, std::forward<overload_t>(overloads)...);
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t, bool auto_launch_op>
auto bind_primitive_registered_operation(
    py::module& module,
    const registered_operation_t<cpp_fully_qualified_name, operation_t, auto_launch_op>& operation) {
    using registered_operation_t = std::decay_t<decltype(operation)>;
    static_assert(
        PrimitiveOperationConcept<operation_t>,
        "Operation must be a primitive operation. Use bind_registered_operation for composite operations");

    using tensor_args_t = typename operation_t::tensor_args_t;
    using operation_attributes_t = typename operation_t::operation_attributes_t;

    using tensor_args_tuple_t = decltype(reflect::to<std::tuple>(std::declval<tensor_args_t>()));
    using operation_attributes_tuple_t = decltype(reflect::to<std::tuple>(std::declval<operation_attributes_t>()));

    auto invoke_lambda = []<class... TensorArgsT, class... OperationAttributesT>(
                             std::type_identity<std::tuple<TensorArgsT...>>,
                             std::type_identity<std::tuple<OperationAttributesT...>>) {
        return [](const registered_operation_t& self,
                  TensorArgsT... tensor_args,
                  OperationAttributesT... operation_attributes,
                  std::uint8_t queue_id = DefaultQueueId) {
            return self(
                queue_id,
                tensor_args_t{std::forward<TensorArgsT>(tensor_args)...},
                operation_attributes_t{std::forward<OperationAttributesT>(operation_attributes)...});
        };
    }(std::type_identity<tensor_args_tuple_t>{}, std::type_identity<operation_attributes_tuple_t>{});

    auto overload =
        [&]<auto... Is, auto... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::array<std::string_view, sizeof...(Is)> tensor_arg_member_names{
                reflect::member_name<Is, tensor_args_t>()...};
            constexpr std::array<std::string_view, sizeof...(Js)> operation_attribute_member_names{
                reflect::member_name<Is, operation_attributes_t>()...};

            return pybind_overload_t{
                invoke_lambda,
                py::arg(std::string{tensor_arg_member_names[Is].data(), tensor_arg_member_names[Is].size()}.c_str())...,
                py::arg(std::string{
                    operation_attribute_member_names[Js].data(), operation_attribute_member_names[Js].size()}
                            .c_str())...,
                py::kw_only(),
                py::arg("queue_id") = 0};
        }(std::make_index_sequence<reflect::size<tensor_args_t>()>{},
          std::make_index_sequence<reflect::size<operation_attributes_t>()>{});

    auto doc_string = fmt::format(
        "{}(tensor_args_tuple, operation_attributes_tuple, queue_id = {}) -> ...",
        operation.base_name(),
        DefaultQueueId);

    return bind_registered_operation_impl(module, operation, doc_string, overload);
}

}  // namespace decorators

using decorators::bind_primitive_registered_operation;
using decorators::bind_registered_operation;
using decorators::pybind_arguments_t;
using decorators::pybind_overload_t;

}  // namespace ttnn
