// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation_concepts.hpp"

namespace ttnn {

// Bind the standard DeviceOperation static methods for descriptor-based ops:
//   - create_output_tensors
//   - compute_output_specs
//   - compute_program_hash (always bound; falls back to default hash when op has no custom impl)
//   - select_program_factory (only if HasSelectProgramFactory<Op>)
template <typename Op>
auto bind_device_op(nanobind::module_& mod, const char* name) {
    namespace nb = nanobind;
    using Params = typename Op::operation_attributes_t;
    using Inputs = typename Op::tensor_args_t;

    auto cls = nb::class_<Op>(mod, name)
                   .def_static(
                       "create_output_tensors",
                       &Op::create_output_tensors,
                       nb::arg("operation_attributes"),
                       nb::arg("tensor_args"))
                   .def_static(
                       "compute_output_specs",
                       &Op::compute_output_specs,
                       nb::arg("operation_attributes"),
                       nb::arg("tensor_args"))
                   .def_static(
                       "compute_program_hash",
                       [](const Params& attrs, const Inputs& tensors) {
                           return ttnn::device_operation::detail::compute_program_hash<Op>(attrs, tensors);
                       },
                       nb::arg("operation_attributes"),
                       nb::arg("tensor_args"));

    if constexpr (ttnn::device_operation::HasSelectProgramFactory<Op>) {
        cls.def_static(
            "select_program_factory",
            &Op::select_program_factory,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"));
    }

    return cls;
}

// Bind a standard 3-arg ProgramDescriptorFactory:
//   create_descriptor(operation_attributes, tensor_args, tensor_return_value) -> ProgramDescriptor
// Types are derived from Op's typedefs.
template <typename F, typename Op>
auto bind_factory(nanobind::module_& mod, const char* name) {
    namespace nb = nanobind;
    using Params = typename Op::operation_attributes_t;
    using Inputs = typename Op::tensor_args_t;
    using Output = typename Op::tensor_return_value_t;

    return nb::class_<F>(mod, name).def_static(
        "create_descriptor",
        [](const Params& p, const Inputs& i, Output& out) { return F::create_descriptor(p, i, out); },
        nb::arg("operation_attributes"),
        nb::arg("tensor_args"),
        nb::arg("tensor_return_value"));
}

}  // namespace ttnn
