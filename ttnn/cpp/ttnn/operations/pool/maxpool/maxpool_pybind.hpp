// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/pool/maxpool/max_pool.hpp"
#include "max_pool2d_pybind.hpp"

#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations {
namespace pool {


void py_module(py::module& module) {
    module.def(
        "max_pool2d",
        &max_pool2d,
        py::arg("input").noconvert(),
        py::arg("in_n").noconvert(),
        py::arg("in_h").noconvert(),
        py::arg("in_w").noconvert(),
        py::arg("kernel_h").noconvert(),
        py::arg("kernel_w").noconvert(),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("pad_h") = 0,
        py::arg("pad_w") = 0,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::kw_only(),
        py::arg("memory_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("nblocks") = 1,
        py::arg("use_multicore") = true,
        R"doc(
        Max Pool 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_n              | Input nbatch                  | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | memory_config     | Output memory config          | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
    )doc");

    module.def(
        "max_pool2d_legacy",
        &max_pool2d_legacy,
        py::arg("input").noconvert(),
        py::arg("reader_indices").noconvert(),
        py::arg("in_n").noconvert(),
        py::arg("in_h").noconvert(),
        py::arg("in_w").noconvert(),
        py::arg("kernel_h").noconvert(),
        py::arg("kernel_w").noconvert(),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("pad_h") = 0,
        py::arg("pad_w") = 0,
        py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1,
        py::kw_only(),
        py::arg("memory_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("nblocks") = 1,
        py::arg("use_multicore") = true,
        R"doc(
        Max Pool 2D
        +-------------------+-------------------------------+---------------+-------------+----------+
        | Argument          | Description                   | Data type     | Valid range | Required |
        +===================+===============================+===============+=============+==========+
        | input             | Input activations tensor      | Tensor        |             | Yes      |
        | in_n              | Input nbatch                  | Tensor        |             | Yes      |
        | in_h              | Input height                  | Tensor        |             | Yes      |
        | in_w              | Input width                   | Tensor        |             | Yes      |
        | kernel_h          | kernel window height          | uint32_t      |             | Yes      |
        | kernel_w          | kernel window width           | uint32_t      |             | Yes      |
        | stride_h          | stride in height dim          | uint32_t      |             | No       |
        | stride_w          | stride in width dim           | uint32_t      |             | No       |
        | pad_h             | padding in height dim         | uint32_t      |             | No       |
        | pad_w             | padding in width dim          | uint32_t      |             | No       |
        | dilation_h        | kernel dilation in height dim | uint32_t      |             | No       |
        | dilation_w        | kernel dilation in width dim  | uint32_t      |             | No       |
        | memory_config     | output tensor memory config   | MemoryConfig  |             | No       |
        +-------------------+-------------------------------+---------------+-------------+----------+
    )doc");

    bind_max_pool2d_operation(module);
}



}  // namespace pool
}  // namespace ttnn::operations
