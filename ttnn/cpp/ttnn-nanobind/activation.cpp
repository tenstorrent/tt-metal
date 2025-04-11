// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "activation.hpp"

#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include "export_enum.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::activation {

namespace nb = nanobind;

void py_module_types(nb::module_& mod) {
    using namespace ttnn::operations::unary;
    export_enum<UnaryOpType>(mod, "UnaryOpType");

    nb::class_<UnaryWithParam>(mod, "UnaryWithParam");
}

void py_module(nb::module_& mod) {
    using namespace ttnn::operations::unary;
    
    // Allow implicit construction of UnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value

    // UnaryWithParam only does floats, so we have to do the cast explicitly

    auto unary_with_param = static_cast<nb::class_<UnaryWithParam>>(mod.attr("UnaryWithParam"));
    unary_with_param
        .def(nb::init<UnaryOpType>())
        .def(nb::init<UnaryOpType, float>())
        .def("__init__", [](UnaryWithParam* t, std::pair<UnaryOpType, float> arg) {
                new (t) UnaryWithParam{arg.first, arg.second};})
        .def("__init__", [](UnaryWithParam* t, std::pair<UnaryOpType, int> arg) {
                new (t) UnaryWithParam{arg.first, static_cast<float>(arg.second)};})
        .def("__init__", [](UnaryWithParam* t, std::pair<UnaryOpType, bool> arg) {
                new (t) UnaryWithParam{arg.first, static_cast<float>(arg.second)};})
        .def_ro("op_type", &UnaryWithParam::op_type)
        .def(nb::init_implicit<UnaryOpType>());
        
    nb::implicitly_convertible<std::pair<UnaryOpType, float>, UnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, int>, UnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, bool>, UnaryWithParam>();

    mod.def("string_to_unary_with_param", &utils::string_to_unary_with_param);
}

}  // namespace ttnn::activation
