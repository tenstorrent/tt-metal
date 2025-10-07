// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "expression_pybind.hpp"
#include "expression.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "ttnn-pybind/export_enum.hpp"

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(ttnn::operations::lazy::Arguments<ttnn::operations::lazy::ExpressionView>);

namespace ttnn::operations::lazy {

template <typename T>
void bind_iterable(py::handle scope, const std::string& name) {
    auto binding =
        py::class_<T>(scope, name.c_str())
            .def(py::init<>())
            .def(py::init<const T&>())
            .def(
                "__getitem__",
                [](const T& iterable, std::ptrdiff_t i) {
                    if (i < 0) {
                        i += iterable.size();
                        if (i < 0) {
                            throw py::index_error();
                        }
                    }
                    auto i_st = static_cast<std::size_t>(i);
                    if (i_st >= iterable.size()) {
                        throw py::index_error();
                    }
                    return iterable[i_st];
                })
            .def(
                "__iter__",
                [](const T& iterable) {
                    return py::make_iterator<py::return_value_policy::copy>(iterable.begin(), iterable.end());
                },
                py::keep_alive<0, 1>())
            .def("__bool__", [](const T& iterable) { return not iterable.empty(); })
            .def("__len__", &T::size);

    using value_type = typename T::value_type;

    if constexpr (requires(T& iter, const value_type& val) { iter.push_back(val); }) {
        binding.def(py::init([](const py::list& list) {
            T result;

            for (const auto& item : list) {
                result.push_back(item.cast<value_type>());
            }

            return result;
        }));

        py::implicitly_convertible<py::list, T>();
    }
}

void py_module(py::module& module) {
    export_enum<Unary>(module);
    export_enum<Binary>(module);
    export_enum<Ternary>(module);

    bind_iterable<Arguments<ExpressionView>>(module, "Arguments");
    bind_iterable<Params>(module, "Params");
    bind_iterable<ParamsView>(module, "ParamsView");

    auto operation = py::class_<Operation>(module, "Operation");
    auto param = py::class_<Param>(module, "Param");
    auto expression_view = py::class_<ExpressionView>(module, "ExpressionView");
    auto function_view = py::class_<FunctionView>(module, "FunctionView");
    auto expression = py::class_<Expression>(module, "Expression");
    auto function = py::class_<Function>(module, "Function");
    auto value = py::class_<Value>(module, "Value");

    expression_view.def(py::init<const ExpressionView&>())
        .def(py::init<const FunctionView&>())
        .def(py::init<const Expression&>())
        .def(py::init<const Function&>())
        .def_property_readonly("tensor", &ExpressionView::tensor)
        .def_property_readonly("function", &ExpressionView::function)
        .def_property_readonly("value", &ExpressionView::value)
        .def_property_readonly("dtype", &ExpressionView::dtype)
        .def_property_readonly("shape", &ExpressionView::logical_shape)
        .def_property_readonly("index", &ExpressionView::index)
        .def_property_readonly("inputs", &ExpressionView::inputs)
        .def_property_readonly("circular_buffers", &ExpressionView::circular_buffers);

    function_view.def(py::init<const FunctionView&>())
        .def(py::init<const Function&>())
        .def_property_readonly("operation", &FunctionView::operation)
        .def_property_readonly("arguments", &FunctionView::arguments)
        .def_property_readonly("params", &FunctionView::params)
        .def_property_readonly("dtype", &FunctionView::dtype)
        .def_property_readonly("shape", &FunctionView::logical_shape)
        .def_property_readonly("index", &FunctionView::index)
        .def_property_readonly("inputs", &FunctionView::inputs)
        .def_property_readonly("circular_buffers", &FunctionView::circular_buffers);

    auto tensor_overload_of = py::overload_cast<const Tensor&>;
    auto unary_overload_of = py::overload_cast<Unary, ExpressionView, Params>;
    auto binary_overload_of = py::overload_cast<Binary, ExpressionView, ExpressionView, Params>;
    auto ternary_overload_of = py::overload_cast<Ternary, ExpressionView, ExpressionView, ExpressionView, Params>;

    expression.def(py::init<const Expression&>())
        .def(py::init<const Function&>())
        .def_property_readonly("tensor", &Expression::tensor)
        .def_property_readonly("function", &Expression::function)
        .def_property_readonly("value", &Expression::value)
        .def_property_readonly("dtype", &Expression::dtype)
        .def_property_readonly("shape", &Expression::logical_shape)
        .def_property_readonly("index", &Expression::index)
        .def_property_readonly("inputs", &Expression::inputs)
        .def_property_readonly("circular_buffers", &Expression::circular_buffers);

    function.def(py::init<const Function&>())
        .def_property_readonly("operation", &Function::operation)
        .def_property_readonly("arguments", &Function::arguments)
        .def_property_readonly("params", &Function::params)
        .def_property_readonly("dtype", &Function::dtype)
        .def_property_readonly("shape", &Function::logical_shape)
        .def_property_readonly("index", &Function::index)
        .def_property_readonly("inputs", &Function::inputs)
        .def_property_readonly("circular_buffers", &Function::circular_buffers);

    py::implicitly_convertible<Expression, ExpressionView>();
    py::implicitly_convertible<Function, FunctionView>();
    py::implicitly_convertible<Function, ExpressionView>();
    py::implicitly_convertible<FunctionView, ExpressionView>();

    module.def("defer", tensor_overload_of(&defer))
        .def("defer", unary_overload_of(&defer))
        .def("defer", binary_overload_of(&defer))
        .def("defer", ternary_overload_of(&defer))
        .def("to_compute_kernel_string", &to_compute_kernel_string)
        .def("to_debug_string", &to_debug_string);
}

}  // namespace ttnn::operations::lazy
