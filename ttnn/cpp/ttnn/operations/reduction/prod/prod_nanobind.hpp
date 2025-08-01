// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"

namespace ttnn::operations::reduction::detail {

template <typename unary_operation_t>
void bind_reduction_prod_operation(nb::module_& mod, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(

            Computes the product of all elements on specified ``dim`` of the ``input`` tensor.

            If no ``dim`` is provided (or ``dim`` is set to `None`), it will compute the product of all elements in the ``input`` tensor.
            If ``keepdim`` is `True`, the resulting tensor will have a similar shape as the ``input`` tensor, but with the specified ``dim`` reduced to 1. This is not supported when taking the product across all dimensions.
            Otherwise, the target ``dim`` will be squeezed, resulting in an output tensor with one less dimension than the ``input`` tensor.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:
                dim (int, optional): Dimension to perform prod. Defaults to `None`.
                keepdim (bool, optional): keep original dimension size. Defaults to `False`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor, dim=0)
                >>> output_all_dims = {1}(tensor)
        )doc",
        operation.base_name(),
        operation.python_fully_qualified_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const std::optional<int64_t> dim,
               const bool keepdim,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, dim, keepdim, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("dim") = std::nullopt,
            nb::arg("keepdim") = false,
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt},
        // prod along nc dimensions
        ttnn::nanobind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const Tensor& output_tensor,
               ttnn::SmallVector<int64_t>& dims,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, output_tensor, dims, memory_config);
            },
            nb::arg("input_tensor"),
            nb::arg("output_tensor"),
            nb::kw_only(),
            nb::arg("dims"),
            nb::arg("memory_config") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::detail
