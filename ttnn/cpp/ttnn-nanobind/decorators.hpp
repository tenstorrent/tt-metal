// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <type_traits>  // decay
#include <tuple>        // forward_as_tuple, apply
#include <utility>      // forward
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

namespace ttnn {
namespace decorators {

// NOLINTBEGIN(performance-unnecessary-value-param)

namespace nb = nanobind;

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

// NOLINTEND(performance-unnecessary-value-param)

}  // namespace decorators

using decorators::nanobind_arguments_t;
using decorators::nanobind_overload_t;

}  // namespace ttnn
