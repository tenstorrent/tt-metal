// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/pool/pool.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace pool {

namespace detail {

void bind_global_avg_pool2d(py::module& module) {
    auto doc = fmt::format(
        R"doc({0}(input_tensor: ttnn.Tensor, *, memory_config: Optional[ttnn.MemoryConfig] = None, dtype: Optional[ttnn.DataType] = None) -> ttnn.Tensor

        Applies {0} to :attr:`input_tensor` by performing a 2D adaptive average pooling over an input signal composed of several input planes. This operation computes the average of all elements in each channel across the entire spatial dimensions.

        .. math::
            {0}(\\mathrm{{input\\_tensor}}_i)

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): The input tensor to be pooled. Typically of shape (batch_size, channels, height, width).

        Keyword Args:
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`dtype` (Optional[ttnn.DataType]): data type for the output tensor

        Returns:
            ttnn.Tensor: The tensor with the averaged values. The output tensor shape is (batch_size, channels, 1, 1).

        Example:

            >>> tensor = ttnn.from_torch(torch.randn((10, 3, 32, 32), dtype=ttnn.bfloat16), device=device)
            >>> output = {1}(tensor)
    )doc",
        ttnn::global_avg_pool2d.base_name(),
        ttnn::global_avg_pool2d.python_fully_qualified_name());

    bind_registered_operation(
        module,
        ttnn::global_avg_pool2d,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt});
}

}  // namespace detail

void py_module(py::module& module) {
    detail::bind_global_avg_pool2d(module);
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
        "max_pool2d_v2",
        &max_pool2d_v2,
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

    module.def(
        "average_pool_2d",
        &average_pool_2d,
        py::arg().noconvert(),
        py::kw_only(),
        py::arg("memory_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("dtype").noconvert() = std::nullopt,
        R"doc(
        Average Pool 2D
        It operates on tensors whose that have channels as the last dimension

        +----------+----------------------------+------------+-------------------------------+----------+
        | Argument | Description                | Data type  | Valid range                   | Required |
        +==========+============================+============+===============================+==========+
        | act      | Input activations tensor   | Tensor     |                               | Yes      |
        +----------+----------------------------+------------+-------------------------------+----------+
    )doc");
}

}  // namespace pool
}  // namespace operations
}  // namespace ttnn
