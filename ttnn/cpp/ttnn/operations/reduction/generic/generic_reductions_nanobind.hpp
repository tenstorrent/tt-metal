// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::reduction::detail {

namespace nb = nanobind;

template <typename reduction_operation_t>
void bind_reduction_operation(nb::module_& mod, const reduction_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Computes the {0} of the input tensor :attr:`input_a` along the specified dimension :attr:`dim`.
        If no dimension is provided, {0} is computed over all dimensions yielding a single value.

        Args:
            input_a (ttnn.Tensor): the input tensor. Must be on the device.
            dim (number): dimension value to reduce over.
            keepdim (bool, optional): keep original dimension size. Defaults to `False`.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            compute_kernel_config (ttnn.ComputeKernelConfig, optional): Compute kernel configuration for the operation. Defaults to `None`.
            scalar (float, optional): A scaling factor to be applied to the input tensor. Defaults to `1.0`.
            correction (bool, optional): Applies only to :func:`ttnn.std` - whether to apply Bessel's correction (i.e. N-1). Defaults to `True`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            The input tensor supports the following data types and layouts:

            .. list-table:: Input Tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - FLOAT32
                  - ROW_MAJOR, TILE
                * - BFLOAT16
                  - ROW_MAJOR, TILE
                * - BFLOAT8_B
                  - ROW_MAJOR, TILE

            The output tensor will match the data type and layout of the input tensor.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded (L1): Width, Height, and ND sharding
            - Output sharding/layout will mirror the input

        Example:
            .. code-block:: python

                input_a = ttnn.rand(1, 2), dtype=torch.bfloat16, device=device)
                output = {1}(input_a, dim, memory_config)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::arg("dim") = nb::none(),
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("scalar") = 1.0f,
            nb::arg("correction") = true});
}

}  // namespace ttnn::operations::reduction::detail
