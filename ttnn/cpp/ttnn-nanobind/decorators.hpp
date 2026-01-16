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

// variants are used for a lot of config types and nanobind otherwise will autogenerate
// the appropriate binding if the typecaster is present. This comes up everywhere
// so include it here. Missing typecasters is a major source of:
// TypeError: __call__(): incompatible function arguments
#include <nanobind/stl/array.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

// used all over the place so just include it here to make the typecaster visible
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace decorators {

// NOLINTBEGIN(performance-unnecessary-value-param)

namespace nb = nanobind;

template <typename T, typename return_t, typename... args_t>
constexpr auto resolve_call_method(return_t (* /*function*/)(args_t...)) {
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

// afuller
template <typename F>
constexpr auto resolve_composite_operation_call_method(F func) {
    using traits = function_traits<F>;

    // Create a new lambda that has the same signature but calls the original function
    return []<typename TSelf, typename... TArgs>(F func, arg_traits<TSelf, TArgs...>) {
        return [func](TSelf self, TArgs... args) ->
               typename traits::return_t { return func(self, static_cast<decltype(args)&&>(args)...); };
    }(func, typename traits::arg_tuple{});
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
    requires device_operation::DeviceOperationConcept<operation_t>
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
    requires(!device_operation::DeviceOperationConcept<operation_t>)
void def_call_operator(py_operation_t& py_operation, const nanobind_overload_t<function_t, py_args_t...>& overload) {
    std::apply(
        [&py_operation, &overload](auto... args) {  // afuller
            // Use composite-specific resolution that preserves the custom lambda logic
            py_operation.def("__call__", resolve_composite_operation_call_method(overload.function), args...);
        },
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
    py_operation.def_prop_ro("__ttnn_operation__", [](const registered_operation_t& /*self*/) { return std::nullopt; });

    (
        [&py_operation](auto&& overload) {
            def_call_operator<registered_operation_t, operation_t>(
                py_operation, std::forward<decltype(overload)>(overload));
        }(std::forward<overload_t>(overloads)),
        ...);

    mod.attr(operation.base_name().c_str()) = operation;  // Bind an instance of the operation to the module

    return py_operation;
}

// NOLINTEND(performance-unnecessary-value-param)

}  // namespace decorators

using decorators::bind_registered_operation;
using decorators::nanobind_arguments_t;
using decorators::nanobind_overload_t;

}  // namespace ttnn
