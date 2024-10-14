// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/maxpool/max_pool2d_pybind.hpp"
#include "ttnn/operations/pool/maxpool/max_pool2d.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/types.hpp"


namespace ttnn::operations::pool {

void bind_max_pool2d_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::max_pool2d,
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
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::max_pool2d)& self, const ttnn::Tensor& input_tensor,
                uint32_t batch_size,
                uint32_t input_h,
                uint32_t input_w,
                uint32_t channels,
                std::array<uint32_t, 2> kernel_size,
                std::array<uint32_t, 2> stride,
                std::array<uint32_t, 2> padding,
                std::array<uint32_t, 2> dilation,
                const uint8_t& queue_id)
                -> ttnn::Tensor { return self(queue_id,
                                            input_tensor,
                                            batch_size,
                                            input_h,
                                            input_w,
                                            channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            dilation); },
                py::arg("input_tensor"),
                py::arg("batch_size"),
                py::arg("input_h"),
                py::arg("input_w"),
                py::arg("channels"),
                py::arg("kernel_size"),
                py::arg("stride"),
                py::arg("padding"),
                py::arg("dilation"),
                py::kw_only(),
                py::arg("queue_id") = 0});
}

void py_module(py::module& module) {
    bind_max_pool2d_operation(module);
}

}  // namespace ttnn::operations::pool
