// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::reduction::detail {

template <typename reduction_operation_t>
void bind_reduction_operation(py::module& module, const reduction_operation_t& operation) {
    namespace py = pybind11;
    auto doc = fmt::format(
        R"doc(
        {0}

        Computes the {0} of the input tensor :attr:`input_a` along the specified dimension :attr:`dim`.
        If no dimension is provided, {0} is computed over all dimensions yielding a single value.

            Args:
                input_a (ttnn.Tensor): the input tensor.
                dim (number): dimension value to reduce over.
                keepdim (bool, optional): keep original dimension size. Defaults to `False`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                compute_kernel_config (ttnn.ComputeKernelConfig, optional): Compute kernel configuration for the operation. Defaults to `None`.
                scalar (float, optional): A scaling factor to be applied to the input tensor. Defaults to `1.0`.
                correction (bool, optional): Applies only to std - whether to apply Bessel's correction (i.e. N-1). Defaults to `True`.

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
                    * - INT32
                        - ROW_MAJOR, TILE
                    * - UINT32
                        - ROW_MAJOR, TILE

                The output tensor will match the data type and layout of the input tensor.

            Example:

                input_a = ttnn.rand(1, 2), dtype=torch.bfloat16, device=device)
                output = {1}(input_a, dim, memory_config)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("dim") = std::nullopt,
            py::arg("keepdim") = false,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("scalar") = 1.0f,
            py::arg("correction") = true});
}

}  // namespace ttnn::operations::reduction::detail
