// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <type_traits> // decay
#include <tuple>       // forward_as_tuple, apply
#include <utility>     // forward
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn/decorators.hpp"
#include "small_vector_caster.hpp"  // NOLINT - for nanobind11 SmallVector binding support.
#include "ttnn/types.hpp"
#include "types.hpp"

namespace ttnn {
namespace decorators {

// NOLINTBEGIN(performance-unnecessary-value-param)

namespace nb = nanobind;

template <typename T, typename return_t, typename... args_t>
constexpr auto resolve_call_method(return_t (*function)(args_t...)) {
    return [](const T& self, args_t... args) { return self(std::forward<args_t>(args)...); };
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
        return [](TSelf self, TArgs... args) ->
               typename traits::return_t { return self(static_cast<decltype(args)&&>(args)...); };
    }(typename traits::arg_tuple{});
}

template <typename... py_args_t>
struct nanobind_arguments_t {
    std::tuple<py_args_t...> value;

    nanobind_arguments_t(py_args_t... args) : value(std::forward_as_tuple(args...)) {}
};

template <typename function_t, typename... py_args_t>
struct nanobind_overload_t {
    function_t function;
    nanobind_arguments_t<py_args_t...> args;

    nanobind_overload_t(function_t function, py_args_t... args) : function{function}, args{args...} {}
};

template <typename registered_operation_t, typename operation_t, typename py_operation_t, typename... py_args_t>
void def_call_operator(py_operation_t& py_operation, const nanobind_arguments_t<py_args_t...>& overload) {
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
    requires PrimitiveOperationConcept<operation_t>
void def_call_operator(py_operation_t& py_operation, const nanobind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) {
            py_operation.def("__call__", resolve_primitive_operation_call_method(overload.function), args...);
        },
        overload.args.value);
}

template <
    typename registered_operation_t,
    typename operation_t,
    typename py_operation_t,
    typename function_t,
    typename... py_args_t>
    requires CompositeOperationConcept<operation_t>
void def_call_operator(py_operation_t& py_operation, const nanobind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) { py_operation.def("__call__", overload.function, args...); },
        overload.args.value);
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t, typename... overload_t>
auto bind_registered_operation(
    nb::module_& mod,
    const registered_operation_t<cpp_fully_qualified_name, operation_t>& operation,
    const std::string& doc,
    overload_t&&... overloads) {
    using registered_operation_t = std::decay_t<decltype(operation)>;
    nb::class_<registered_operation_t> py_operation(mod, operation.class_name().c_str());

    py_operation.doc() = doc;

    py_operation.def_prop_ro(
        "name",
        [](const registered_operation_t& self) -> std::string { return self.base_name(); },
        "Shortened name of the api");

    py_operation.def_prop_ro(
        "python_fully_qualified_name",
        [](const registered_operation_t& self) -> std::string { return self.python_fully_qualified_name(); },
        "Fully qualified name of the api");

    // Attribute to identify of ttnn operations
    py_operation.def_prop_ro(
        "__ttnn_operation__", [](const registered_operation_t& self) { return std::nullopt; });

    py_operation.def_prop_ro(
        "is_primitive",
        [](const registered_operation_t& self) -> bool { return registered_operation_t::is_primitive; },
        "Specifies if the operation maps to a single program");

    (
        [&py_operation](auto&& overload) {
            def_call_operator<registered_operation_t, operation_t>(
                py_operation, std::forward<decltype(overload)>(overload));
        }(std::forward<overload_t>(overloads)),
        ...);

    // Bind an instance of the operation to the module (not the class type).
    // Using a reference ensures Python gets an instance, avoiding passing the class
    // object as 'self' to __call__ (which leads to signature errors).
    mod.attr(operation.base_name().c_str()) = nb::cast(operation, nb::rv_policy::reference);

    return py_operation;
}

// NOLINTEND(performance-unnecessary-value-param)

}  // namespace decorators

using decorators::bind_registered_operation;
using decorators::nanobind_arguments_t;
using decorators::nanobind_overload_t;

}  // namespace ttnn
