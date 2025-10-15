// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "expression_pybind.hpp"
#include "lazy.hpp"

#include <ttnn-pybind/export_enum.hpp>
#include <tt_stl/type_name.hpp>

#include <pybind11/pybind11.h>
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
void def_overloads(py::class_<Func>& cls, OverloadsFor<Derived, Args...>) {
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
void def_overloads(py::class_<Func>& cls, ttsl::overloaded<Overloads...> functor) {
    (..., def_overloads(cls, static_cast<Overloads>(functor)));
}

template <typename Func>
void def_functor(py::handle scope, const std::string& name, Func functor) {
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

    def_functor(module, "recip", recip);
    def_functor(module, "negative", negative);
    def_functor(module, "exp", exp);
    def_functor(module, "eqz", eqz);
    def_functor(module, "gez", gez);
    def_functor(module, "gtz", gtz);
    def_functor(module, "lez", lez);
    def_functor(module, "ltz", ltz);
    def_functor(module, "nez", nez);
    def_functor(module, "logical_not", logical_not);
    def_functor(module, "atan", atan);
    def_functor(module, "eq", eq);
    def_functor(module, "ge", ge);
    def_functor(module, "gt", gt);
    def_functor(module, "le", le);
    def_functor(module, "lt", lt);
    def_functor(module, "ne", ne);
    def_functor(module, "add", add);
    def_functor(module, "sub", sub);
    def_functor(module, "rsub", rsub);
    def_functor(module, "mul", mul);
    def_functor(module, "pow", pow);
    def_functor(module, "rpow", rpow);
    def_functor(module, "div", div);
    def_functor(module, "rdiv", rdiv);
    def_functor(module, "logical_and", logical_and);
    def_functor(module, "logical_or", logical_or);
    def_functor(module, "logical_xor", logical_xor);
    def_functor(module, "where", where);
    def_functor(module, "atan2", atan2);
}

}  // namespace ttnn::operations::lazy
