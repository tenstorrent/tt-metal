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
        R"doc({0}(input_tensor: ttnn.Tensor, all_dimensions: bool, dim: int *, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Computes the prod function along specified ``dim`` or all dimensions on the ``input`` tensor.

            .. math::
                {0}(\\mathrm{{input\\_tensor}}_i)

            Args:
                * :attr:`input_tensor`
                * :attr:`all_dimensions` - If ``all_dimensions`` is set to ``true`` irrespective of given dimension it will prod along all dimensions. Default value = False.
                * :attr:`dim` - Dimension to perform prod, Default value = 0.

            Keyword Args:
                * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.


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
