// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "activation.hpp"

#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>

#include "export_enum.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::activation {

void py_module_types(nb::module_& mod) {
    using namespace ttnn::operations::unary;
    export_enum<UnaryOpType>(mod, "UnaryOpType");

    nb::class_<UnaryWithParam>(mod, "UnaryWithParam");
    nb::class_<EltwiseUnaryWithParam>(mod, "EltwiseUnaryWithParam");
}

void py_module(nb::module_& mod) {
    using namespace ttnn::operations::unary;

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

    // Allow implicit construction of UnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value
    nb::implicitly_convertible<UnaryOpType, UnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, float>, UnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, int>, UnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, bool>, UnaryWithParam>();

    mod.def("string_to_unary_with_param", &utils::string_to_unary_with_param);

    // TODO_NANOBIND: finish conversion with placement new
    auto eltwise_unary_with_param = static_cast<nb::class_<EltwiseUnaryWithParam>>(mod.attr("EltwiseUnaryWithParam"));
    eltwise_unary_with_param.def(nb::init<UnaryOpType>())
        .def(nb::init<UnaryOpType, float>())
        .def(nb::init<UnaryOpType, int32_t>())
        .def(nb::init<UnaryOpType, uint32_t>())
        .def(nb::init<UnaryWithParam>())
        .def(
            "__init__",
            [](EltwiseUnaryWithParam* t, std::pair<UnaryOpType, float> arg) {
                new (t) EltwiseUnaryWithParam{arg.first, arg.second};
            })
        .def(
            "__init__",
            [](EltwiseUnaryWithParam* t, std::pair<UnaryOpType, int32_t> arg) {
                new (t) EltwiseUnaryWithParam{arg.first, static_cast<float>(arg.second)};
            })
        .def(
            "__init__",
            [](EltwiseUnaryWithParam* t, std::pair<UnaryOpType, uint32_t> arg) {
                new (t) EltwiseUnaryWithParam{arg.first, static_cast<float>(arg.second)};
            })
        //.def(nb::init<>([](std::pair<UnaryOpType, float> arg) { return EltwiseUnaryWithParam{arg.first, arg.second};
        //})) .def(nb::init<>([](std::pair<UnaryOpType, int> arg) { return EltwiseUnaryWithParam{arg.first, arg.second};
        //}))
        .def_prop_ro("op_type", &EltwiseUnaryWithParam::type);

    // Allow implicit construction of EltwiseUnaryWithParam object without user explicitly creating it
    // Can take in just the op type, or sequence container of op type and param value
    nb::implicitly_convertible<UnaryOpType, EltwiseUnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, float>, EltwiseUnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, int32_t>, EltwiseUnaryWithParam>();
    nb::implicitly_convertible<std::pair<UnaryOpType, uint32_t>, EltwiseUnaryWithParam>();
    nb::implicitly_convertible<UnaryWithParam, EltwiseUnaryWithParam>();
}

// NOLINTEND(bugprone-unused-raii)

}  // namespace ttnn::activation
