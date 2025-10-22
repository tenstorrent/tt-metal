// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "expression_pybind.hpp"
#include "lazy.hpp"

#include <ttnn-pybind/export_enum.hpp>
#include <tt_stl/type_name.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

PYBIND11_MAKE_OPAQUE(ttnn::operations::lazy::Arguments<ttnn::operations::lazy::ExpressionView>);

namespace ttnn::operations::lazy {

template <typename T>
void bind_iterable(py::handle scope, const std::string& name) {
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
}

template <typename... Args, typename Func, typename... Extra>
auto def_overload(py::class_<Func>& cls, const Extra&... extra) {
    if constexpr (requires { py::overload_cast<Args...>(&Func::operator(), py::const_); }) {
        cls.def("__call__", py::overload_cast<Args...>(&Func::operator(), py::const_), extra...);
    }
}

template <typename Func, typename Derived, typename... Args>
void def_overloads(py::class_<Func>& cls, const OverloadsFor<Derived, Args...>&) {
    using mp_pybind_map = mp::mp_list<
        // also bind const Tensor& wherever ExpressionView is bound
        mp::mp_list<ExpressionView, const Tensor&>
        // additional binding map elements go here
        >;

    using mp_overloads = mp::mp_product<mp::mp_list, mp_find_from<Args, mp_pybind_map>...>;

    mp::mp_for_each<mp_overloads>(ttsl::overloaded{
        [&]<typename First>(mp::mp_list<First>) { def_overload<First>(cls, py::arg("first")); },
        [&]<typename First, typename Second>(mp::mp_list<First, Second>) {
            def_overload<First, Second>(cls, py::arg("first"), py::arg("second"));
        },
        [&]<typename First, typename Second, typename Third>(mp::mp_list<First, Second, Third>) {
            def_overload<First, Second, Third>(cls, py::arg("first"), py::arg("second"), py::arg("third"));
        },
    });
}

template <typename Func, typename... Overloads>
void def_overloads(py::class_<Func>& cls, const ttsl::overloaded<Overloads...>& functor) {
    (..., def_overloads(cls, static_cast<const Overloads&>(functor)));
}

template <typename Func>
void def_functor(py::handle scope, const std::string& name, const Func& functor) {
    static_assert(
        not ttsl::short_type_name<Func>.ends_with('>'),
        "Func must be a non-template strong type like OverloadedBinaryFn, not a template specialization like "
        "overloaded<...>");
    const std::string type_name{ttsl::short_type_name<Func>};

    if (not py::hasattr(scope, type_name.c_str())) {
        auto cls = py::class_<Func>(scope, type_name.c_str());
        def_overloads(cls, functor);
    }

    scope.attr(name.c_str()) = functor;
}

template <typename Func, typename Operation>
void def_rbinary(py::class_<Func>& cls, const std::string& name, const Operation& operation) {
    const auto with_tensor = [=](const Func& first, const Tensor& second) { return operation(first, second); };
    const auto with_param = [=](const Func& first, Param second) { return operation(first, second); };

    cls.def(name.c_str(), with_tensor, py::is_operator());
    cls.def(name.c_str(), with_param, py::is_operator());
}

template <typename Func, typename Operation>
void def_binary(py::class_<Func>& cls, const std::string& name, const Operation& operation) {
    const auto with_expression_view = [=](const Func& first, ExpressionView second) {
        return operation(first, second);
    };

    cls.def(name.c_str(), with_expression_view, py::is_operator());
    def_rbinary(cls, name, operation);
}

void py_module(py::module& module) {
    export_enum<Unary>(module);
    export_enum<UnaryWithParam>(module);
    export_enum<Binary>(module);
    export_enum<Ternary>(module);

    bind_iterable<Arguments<ExpressionView>>(module, "Arguments");
    bind_iterable<ParamsView>(module, "ParamsView");

    auto operation = py::class_<Operation>(module, "Operation");
    auto param = py::class_<Param>(module, "Param");
    auto expression_view = py::class_<ExpressionView>(module, "ExpressionView");
    auto function_view = py::class_<FunctionView>(module, "FunctionView");
    auto expression = py::class_<Expression>(module, "Expression");
    auto function = py::class_<Function>(module, "Function");
    auto value = py::class_<Value>(module, "Value");

    py::implicitly_convertible<Expression, ExpressionView>();
    py::implicitly_convertible<Function, FunctionView>();
    py::implicitly_convertible<Function, ExpressionView>();
    py::implicitly_convertible<FunctionView, ExpressionView>();

    expression_view.def(py::init<const ExpressionView&>())
        .def(py::init<const FunctionView&>())
        .def(py::init<const Expression&>())
        .def(py::init<const Function&>())
        .def(-py::self)
        .def("tensor", &ExpressionView::tensor)
        .def("function", &ExpressionView::function)
        .def("value", &ExpressionView::value)
        .def("dtype", &ExpressionView::dtype)
        .def("shape", &ExpressionView::logical_shape)
        .def("cb_index", &ExpressionView::cb_index)
        .def("rt_offset", &ExpressionView::rt_offset)
        .def("inputs", &ExpressionView::inputs)
        .def("circular_buffers", &ExpressionView::circular_buffers);

    function_view.def(py::init<const FunctionView&>())
        .def(py::init<const Function&>())
        .def(-py::self)
        .def("__str__", &to_debug_string)
        .def("source", &to_compute_kernel_string)
        .def("operation", &FunctionView::operation)
        .def("arguments", &FunctionView::arguments)
        .def("params", &FunctionView::params)
        .def("dtype", &FunctionView::dtype)
        .def("shape", &FunctionView::logical_shape)
        .def("cb_index", &FunctionView::cb_index)
        .def("rt_offset", &FunctionView::rt_offset)
        .def("inputs", &FunctionView::inputs)
        .def("circular_buffers", &FunctionView::circular_buffers);

    expression.def(py::init<const Expression&>())
        .def(py::init<const Function&>())
        .def(-py::self)
        .def("tensor", &Expression::tensor)
        .def("function", &Expression::function)
        .def("value", &Expression::value)
        .def("dtype", &Expression::dtype)
        .def("shape", &Expression::logical_shape)
        .def("cb_index", &Expression::cb_index)
        .def("rt_offset", &Expression::rt_offset)
        .def("inputs", &Expression::inputs)
        .def("circular_buffers", &Expression::circular_buffers);

    function.def(py::init<const Function&>())
        .def(-py::self)
        .def("__str__", &to_debug_string)
        .def("source", &to_compute_kernel_string)
        .def("operation", &Function::operation)
        .def("arguments", &Function::arguments)
        .def("params", &Function::params)
        .def("dtype", &Function::dtype)
        .def("shape", &Function::logical_shape)
        .def("cb_index", &Function::cb_index)
        .def("rt_offset", &Function::rt_offset)
        .def("inputs", &Function::inputs)
        .def("circular_buffers", &Function::circular_buffers);

    const auto def_all_binary = [&](const std::string& name, auto operation) {
        def_binary(expression_view, name, operation);
        def_binary(function_view, name, operation);
        def_binary(expression, name, operation);
        def_binary(function, name, operation);
    };

    const auto def_all_rbinary = [&](const std::string& name, auto operation) {
        def_rbinary(expression_view, name, operation);
        def_rbinary(function_view, name, operation);
        def_rbinary(expression, name, operation);
        def_rbinary(function, name, operation);
    };

    def_all_binary("__add__", lazy::add);
    def_all_rbinary("__radd__", lazy::add);

    def_all_binary("__sub__", lazy::sub);

    def_all_binary("__mul__", lazy::mul);
    def_all_rbinary("__rmul__", lazy::mul);

    def_all_binary("__truediv__", lazy::div);

    def_all_binary("__lt__", lazy::lt);
    def_all_binary("__gt__", lazy::gt);
    def_all_binary("__eq__", lazy::eq);
    def_all_binary("__ne__", lazy::ne);
    def_all_binary("__le__", lazy::le);
    def_all_binary("__ge__", lazy::ge);

    def_all_binary("__and__", lazy::logical_and);
    def_all_rbinary("__rand__", lazy::logical_and);

    def_all_binary("__or__", lazy::logical_or);
    def_all_rbinary("__ror__", lazy::logical_or);

    def_functor(module, "reciprocal", lazy::reciprocal);
    def_functor(module, "neg", lazy::neg);
    def_functor(module, "exp", lazy::exp);
    def_functor(module, "eqz", lazy::eqz);
    def_functor(module, "gez", lazy::gez);
    def_functor(module, "gtz", lazy::gtz);
    def_functor(module, "lez", lazy::lez);
    def_functor(module, "ltz", lazy::ltz);
    def_functor(module, "nez", lazy::nez);
    def_functor(module, "logical_not", lazy::logical_not);
    def_functor(module, "atan", lazy::atan);
    def_functor(module, "eq", lazy::eq);
    def_functor(module, "ge", lazy::ge);
    def_functor(module, "gt", lazy::gt);
    def_functor(module, "le", lazy::le);
    def_functor(module, "lt", lazy::lt);
    def_functor(module, "ne", lazy::ne);
    def_functor(module, "add", lazy::add);
    def_functor(module, "sub", lazy::sub);
    def_functor(module, "mul", lazy::mul);
    def_functor(module, "pow", lazy::pow);
    def_functor(module, "div", lazy::div);
    def_functor(module, "logical_and", lazy::logical_and);
    def_functor(module, "logical_or", lazy::logical_or);
    def_functor(module, "logical_xor", lazy::logical_xor);
    def_functor(module, "where", lazy::where);
    def_functor(module, "atan2", lazy::atan2);
}

}  // namespace ttnn::operations::lazy
