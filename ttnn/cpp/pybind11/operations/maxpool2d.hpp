// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/operations/maxpool2d.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::maxpool {

using array2_t = std::array<uint32_t, 2>;

void py_module(py::module& module) {
    module.def(
        "maxpool2d",
        [](const ttnn::Tensor& input_tensor,
            uint32_t batch_size,
            uint32_t input_height,
            uint32_t input_width,
            uint32_t channels,
            array2_t kernel_size,
            array2_t stride,
            array2_t padding,
            array2_t dilation,
            Device& device) -> Tensor {
            return maxpool2d(input_tensor, batch_size, input_height, input_width, channels, kernel_size, stride, padding, dilation, device);
        },
        py::kw_only(),
        py::arg("input_tensor"),
        py::arg("batch_size"),
        py::arg("input_height"),
        py::arg("input_width"),
        py::arg("channels"),
        py::arg("kernel_size"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("device"));
}

}  // namespace ttnn::operations::maxpool
