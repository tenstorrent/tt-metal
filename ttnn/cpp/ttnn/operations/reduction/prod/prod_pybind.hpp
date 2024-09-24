// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "ttnn/operations/reduction/prod/prod.hpp"

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;

template <typename unary_operation_t>
void bind_reduction_prod_operation(py::module& module, const unary_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(

            Computes the prod function along specified ``dim`` or all dimensions on the ``input`` tensor.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                all_dimensions (bool, optional): prod along all dimensions. Defaults to `False`.
                dim (int, optional): Dimension to perform prod. Defaults to `0`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.

            Example::

                >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
                >>> output = {1}(tensor)
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
               bool all_dimensions,
               int dim,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, all_dimensions, dim, memory_config); },
            py::arg("input_tensor"),
            py::arg("all_dimensions") = false,
            py::arg("dim") = 0,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt},
        // prod along nc dimensions
        ttnn::pybind_overload_t{
            [](const unary_operation_t& self,
               const Tensor& input_tensor,
               const Tensor& output_tensor,
               std::vector<int64_t> &dims,
               const std::optional<MemoryConfig>& memory_config) {
                return self(input_tensor, output_tensor, dims, memory_config); },
            py::arg("input_tensor"),
            py::arg("output_tensor"),
            py::kw_only(),
            py::arg("dims") = std::vector<int64_t>(),
            py::arg("memory_config") = std::nullopt}
            );
}

}  // namespace ttnn::operations::reduction::detail
