// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "export_enum.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace py = pybind11;

namespace ttnn {
namespace activation {

void py_module_types(py::module& m) {
    using namespace ttnn::operations::unary;
    export_enum<UnaryOpType>(m, "UnaryOpType");

    py::class_<UnaryWithParam>(m, "UnaryWithParam");
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
}

}  // namespace activation
}  // namespace ttnn
