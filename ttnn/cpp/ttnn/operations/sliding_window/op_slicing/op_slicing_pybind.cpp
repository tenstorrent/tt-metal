// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace tt::tt_metal;

namespace ttnn::operations::op_slicing {

void py_bind_sliding_window(py::module& module) {
    module.def(
        "run_sliced_op",
        &run_sliced_op,
        py::arg("input_tensor"),
        py::arg("output_tensor"),
        py::arg("op_slice_attr"),
        py::arg("dram_slice_config"),
        DOC("Runs a 2D operation in slices based on the provided slicing configuration and operation attributes."));
}

}  // namespace ttnn::operations::op_slicing
