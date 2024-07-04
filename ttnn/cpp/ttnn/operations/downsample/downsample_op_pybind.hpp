// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device/downsample_op.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace downsample {

void bind_downsample(py::module& module) {
    const auto doc = R"doc(
 Downsamples a given multi-channel 2D (spatial) data.
 The input data is assumed to be of the form [N, H, W, C].

 Args:
     * :attr:`input_tensor`: the input tensor
     * :attr:`downsample_params`: Params list: batch size, conv input H, conv input W, conv stride H, conv stride W
     )doc";

    ttnn::bind_registered_operation(
        module,
        ttnn::downsample,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("downsample_params"),
            py::arg("output_dtype") = std::nullopt});
}
void py_module(py::module& module) {
    bind_downsample(module);
}

}
}
}
