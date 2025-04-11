// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/pool/global_avg_pool/global_avg_pool.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::avgpool {

namespace {

void bind_global_avg_pool2d(nb::module_& mod) {
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
        mod,
        ttnn::global_avg_pool2d,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none()});
}

}  // namespace

void py_module(nb::module_& mod) { bind_global_avg_pool2d(mod); }

}  // namespace ttnn::operations::avgpool
