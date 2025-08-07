// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/reduction/prod/prod.hpp"

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;

template <typename unary_operation_t>
void bind_reduction_prod_operation(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(

            Computes the product of all elements on specified ``dim`` of the ``input`` tensor.

            If no ``dim`` is provided (or ``dim`` is set to `None`), it will compute the full product of every element in the ``input`` tensor.
            When using this full-product mode, the input tensor must be bfloat16.

            If ``keepdim`` is `True`, the resulting tensor will have a similar shape as the ``input`` tensor, but with the specified ``dim`` reduced to 1.
            Otherwise, the target ``dim`` will be squeezed, resulting in an output tensor with one less dimension than the ``input`` tensor.
            Setting ``keepdim`` to `True` is not supported when computing the full product, as this operation results in a scalar.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:
                dim (int, optional): Dimension to perform prod. Defaults to `None`.
                keepdim (bool, optional): keep original dimension size. Defaults to `False`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.

            Example::
                tensor = ttnn.rand((1,2), device=device)
                output = {1}(tensor, dim=0)
                output_all_dims = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<int64_t> dim,
               const bool keepdim,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, dim, keepdim, memory_config);
            },
            py::arg("input_tensor"),
            py::arg("dim") = std::nullopt,
            py::arg("keepdim") = false,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},
        // prod along nc dimensions
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const Tensor& output_tensor,
               ttnn::SmallVector<int64_t>& dims,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, output_tensor, dims, memory_config);
            },
            py::arg("input_tensor"),
            py::arg("output_tensor"),
            py::kw_only(),
            py::arg("dims"),
            py::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::detail
