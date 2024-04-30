// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn/operations/others.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace others {

void py_module(py::module& module) {

    module.def("upsample", &upsample,
        py::arg("input_tensor"),
        py::arg("scale_factor"),
        py::arg("memory_config") = std::nullopt,
        R"doc(
Upsamples a given multi-channel 2D (spatial) data.
The input data is assumed to be of the form [N, H, W, C].

The algorithms available for upsampling are 'nearest' for now.

Args:
    * :attr:`input_tensor`: the input tensor
    * :attr:`scale_factor`: multiplier for spatial size. Has to match input size if it is a tuple.
    )doc");

}

}  // namespace normalization
}  // namespace operations
}  // namespace ttnn
