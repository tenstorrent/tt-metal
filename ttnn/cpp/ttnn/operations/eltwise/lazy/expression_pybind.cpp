// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "expression_pybind.hpp"
#include "expression.hpp"
#include "lazy.hpp"
#include <ttnn-pybind/export_enum.hpp>
#include <tt_stl/type_name.hpp>

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

template <typename... Args, typename Func, typename... Extra>
auto def_overload(py::class_<Func>& cls, const Extra&... extra) {
    if constexpr (requires { py::overload_cast<Args...>(&Func::operator(), py::const_); }) {
        cls.def("__call__", py::overload_cast<Args...>(&Func::operator(), py::const_), extra...);
    }
}

template <typename Func>
void def_overloads(py::handle scope, Func functor, const std::string& name) {
    std::string type_name{ttsl::short_type_name<Func>};
    std::erase_if(type_name, [](unsigned char ch) -> bool { return not std::isalnum(ch); });

    if (not py::hasattr(scope, type_name.c_str())) {
        auto cls = py::class_<Func>(scope, type_name.c_str());

        const auto def_unary = [&]<typename... First>(mp::mp_list<mp::mp_list<First>...>) {
            (..., def_overload<First>(cls, py::arg("first")));
        };
        const auto def_binary = [&]<typename... First, typename... Second>(mp::mp_list<mp::mp_list<First, Second>...>) {
            (..., def_overload<First, Second>(cls, py::arg("first"), py::arg("second")));
        };
        const auto def_ternary = [&]<typename... First, typename... Second, typename... Third>(
                                     mp::mp_list<mp::mp_list<First, Second, Third>...>) {
            (..., def_overload<First, Second, Third>(cls, py::arg("first"), py::arg("second"), py::arg("third")));
        };

        using Arg = mp::mp_apply<mp::mp_set_union, mp_convert_map>;
        using TensorArg = mp::mp_list<ExpressionView, const Tensor&>;

        def_unary(mp::mp_product<mp::mp_list, Arg>{});
        def_binary(mp::mp_product<mp::mp_list, Arg, Arg>{});
        // replace TensorArg with Arg below when UnaryWithParam2 or BinaryWithParam operations are added
        // TensorArg binds 2*2*2=8 overloads, but Arg will bind 6*6*6=216 overloads
        def_ternary(mp::mp_product<mp::mp_list, TensorArg, TensorArg, TensorArg>{});
    }

    scope.attr(name.c_str()) = py::cast(functor);
}

void py_module(py::module& module) {
    export_enum<Unary>(module);
    export_enum<UnaryWithParam>(module);
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
        .def_property_readonly("cb_index", &ExpressionView::cb_index)
        .def_property_readonly("rt_offset", &ExpressionView::rt_offset)
        .def_property_readonly("inputs", &ExpressionView::inputs)
        .def_property_readonly("circular_buffers", &ExpressionView::circular_buffers);

    function_view.def(py::init<const FunctionView&>())
        .def(py::init<const Function&>())
        .def_property_readonly("operation", &FunctionView::operation)
        .def_property_readonly("arguments", &FunctionView::arguments)
        .def_property_readonly("params", &FunctionView::params)
        .def_property_readonly("dtype", &FunctionView::dtype)
        .def_property_readonly("shape", &FunctionView::logical_shape)
        .def_property_readonly("cb_index", &FunctionView::cb_index)
        .def_property_readonly("rt_offset", &FunctionView::rt_offset)
        .def_property_readonly("inputs", &FunctionView::inputs)
        .def_property_readonly("circular_buffers", &FunctionView::circular_buffers);

    expression.def(py::init<const Expression&>())
        .def(py::init<const Function&>())
        .def_property_readonly("tensor", &Expression::tensor)
        .def_property_readonly("function", &Expression::function)
        .def_property_readonly("value", &Expression::value)
        .def_property_readonly("dtype", &Expression::dtype)
        .def_property_readonly("shape", &Expression::logical_shape)
        .def_property_readonly("cb_index", &Expression::cb_index)
        .def_property_readonly("rt_offset", &Expression::rt_offset)
        .def_property_readonly("inputs", &Expression::inputs)
        .def_property_readonly("circular_buffers", &Expression::circular_buffers);

    function.def(py::init<const Function&>())
        .def_property_readonly("operation", &Function::operation)
        .def_property_readonly("arguments", &Function::arguments)
        .def_property_readonly("params", &Function::params)
        .def_property_readonly("dtype", &Function::dtype)
        .def_property_readonly("shape", &Function::logical_shape)
        .def_property_readonly("cb_index", &Function::cb_index)
        .def_property_readonly("rt_offset", &Function::rt_offset)
        .def_property_readonly("inputs", &Function::inputs)
        .def_property_readonly("circular_buffers", &Function::circular_buffers);

    py::implicitly_convertible<Expression, ExpressionView>();
    py::implicitly_convertible<Function, FunctionView>();
    py::implicitly_convertible<Function, ExpressionView>();
    py::implicitly_convertible<FunctionView, ExpressionView>();

    module.def("to_compute_kernel_string", &to_compute_kernel_string).def("to_debug_string", &to_debug_string);

    def_overloads(module, recip, "recip");
    def_overloads(module, negative, "negative");
    def_overloads(module, exp, "exp");
    def_overloads(module, eqz, "eqz");
    def_overloads(module, gez, "gez");
    def_overloads(module, gtz, "gtz");
    def_overloads(module, lez, "lez");
    def_overloads(module, ltz, "ltz");
    def_overloads(module, nez, "nez");
    def_overloads(module, logical_not, "logical_not");
    def_overloads(module, eq, "eq");
    def_overloads(module, ge, "ge");
    def_overloads(module, gt, "gt");
    def_overloads(module, le, "le");
    def_overloads(module, lt, "lt");
    def_overloads(module, ne, "ne");
    def_overloads(module, add, "add");
    def_overloads(module, sub, "sub");
    def_overloads(module, rsub, "rsub");
    def_overloads(module, mul, "mul");
    def_overloads(module, div, "div");
    def_overloads(module, rdiv, "rdiv");
    def_overloads(module, power, "power");
    def_overloads(module, where, "where");
}

}  // namespace ttnn::operations::lazy
