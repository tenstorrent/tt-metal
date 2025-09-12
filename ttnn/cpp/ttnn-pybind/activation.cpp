// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "activation.hpp"

#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "export_enum.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::activation {

void py_module_types(py::module& m) {
    using namespace ttnn::operations::unary;
    export_enum<UnaryOpType>(m, "UnaryOpType");

    py::class_<UnaryWithParam>(m, "UnaryWithParam");
    py::class_<EltwiseUnaryWithParam>(m, "EltwiseUnaryWithParam");
}

void py_module(py::module& m) {
    using namespace ttnn::operations::unary;

    auto unary_with_param = static_cast<py::class_<UnaryWithParam>>(m.attr("UnaryWithParam"));
    unary_with_param.def(py::init<UnaryOpType>())
        .def(py::init<UnaryOpType, float>())
        .def(py::init<>([](std::pair<UnaryOpType, float> arg) { return UnaryWithParam{arg.first, arg.second}; }))
        .def_readonly("op_type", &UnaryWithParam::op_type);

    // Allow implicit construction of UnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value
    py::implicitly_convertible<UnaryOpType, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, float>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, int>, UnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, bool>, UnaryWithParam>();

    m.def("string_to_unary_with_param", &utils::string_to_unary_with_param);

    auto eltwise_unary_with_param = static_cast<py::class_<EltwiseUnaryWithParam>>(m.attr("EltwiseUnaryWithParam"));
    eltwise_unary_with_param.def(py::init<UnaryOpType>())
        .def(py::init<UnaryOpType, float>())
        .def(py::init<UnaryOpType, int>())
        .def(py::init<UnaryWithParam>())
        .def(py::init<>([](std::pair<UnaryOpType, float> arg) { return EltwiseUnaryWithParam{arg.first, arg.second}; }))
        .def(py::init<>([](std::pair<UnaryOpType, int> arg) { return EltwiseUnaryWithParam{arg.first, arg.second}; }))
        .def_property_readonly("op_type", &EltwiseUnaryWithParam::type);

    // Allow implicit construction of EltwiseUnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value
    py::implicitly_convertible<UnaryOpType, EltwiseUnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, float>, EltwiseUnaryWithParam>();
    py::implicitly_convertible<std::pair<UnaryOpType, int>, EltwiseUnaryWithParam>();
    py::implicitly_convertible<UnaryWithParam, EltwiseUnaryWithParam>();
}

}  // namespace ttnn::activation
