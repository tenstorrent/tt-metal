// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/pool/avgpool/avg_pool.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;
namespace ttnn {
namespace operations {
namespace avgpool {

namespace detail {

void bind_global_avg_pool2d(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Applies {0} to :attr:`input_tensor` by performing a 2D adaptive average pooling over an input signal composed of several input planes. This operation computes the average of all elements in each channel across the entire spatial dimensions.

        .. math::
            {0}(\\mathrm{{input\\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor. Typically of shape (batch_size, channels, height, width).


        Keyword args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`


        Returns:
            ttnn.Tensor: the output tensor with the averaged values. The output tensor shape is (batch_size, channels, 1, 1).


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
        "avg_pool2d",
        &avg_pool2d,
        py::arg().noconvert(),
        py::kw_only(),
        py::arg("memory_config").noconvert() = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        py::arg("dtype").noconvert() = std::nullopt,
        R"doc(
        Average Pool 2D
        It operates on tensors that have channels as the last dimension.

        +----------+----------------------------+------------+-------------------------------+----------+
        | Argument | Description                | Data type  | Valid range                   | Required |
        +==========+============================+============+===============================+==========+
        | act      | Input activations tensor   | Tensor     |                               | Yes      |
        +----------+----------------------------+------------+-------------------------------+----------+
    )doc");
}

}  // namespace avgpool
}  // namespace operations
}  // namespace ttnn
