// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "op_slicing.hpp"
#include "op_slicing_pybind.hpp"
#include <pybind11/cast.h>
#include "ttnn-pybind/decorators.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::op_slicing {

void py_bind_op_slicing(py::module& module) {
    module.def(
        "run_sliced_op",
        &run_sliced_op,
        py::kw_only(),
        py::arg("input_tensor"),
        py::arg("output_tensor"),
        py::arg("op_slice_attr"),
        py::arg("dram_slice_config"),
        R"doc(
        Applies a 2D convolution over an input signal composed of several input planes.

        For more information, refer to `this tech report. <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md>`
    )doc");

    py::class_<OpSliceAttr> py_op_slice_attr(
        module,
        "OpSliceAttr",
        R"doc(
        OpSliceAttr is an interface that defines how to slice the input tensor based on the output slice.
        )doc");
}

}  // namespace ttnn::operations::op_slicing
