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
            Computes the product of all elements on specified :attr:`dim` of the :attr:`input_tensor` tensor.

            If no :attr:`dim` is provided (or :attr:`dim` is set to `None`), it will compute the full product of every element in the :attr:`input_tensor` tensor.

            If :attr:`keepdim` is `True`, the resulting tensor will have the same rank as the :attr:`input_tensor` tensor, but with the specified :attr:`dim` reduced to 1.
            Otherwise, the target :attr:`dim` will be squeezed, resulting in an output tensor with one less dimension than the :attr:`input_tensor` tensor.


            {0} is also overloaded with a niche NC version of this function, with the following definition:

            ``{1}(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor``

            This version allows for a list of :attr:`dims` to be specified instead of a :attr:`dim`, requires an :attr:`output_tensor` tensor, and does not support :attr:`keepdim`.
            It is only intended for use with the NC dimensions (0, 1).

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:
                dim (int, optional): Dimension to perform prod. Defaults to `None`.
                dims (List[int], optional): Dimensions to perform prod. Defaults to `None`. Mutually exclusive with `dim`.
                keepdim (bool, optional): keep original dimension size. Defaults to `False`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.

            Note:
                The :attr:`input_tensor` supports the following data type and layout:

                .. list-table:: input_tensor
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16
                      - TILE, ROW_MAJOR

                The :attr:`output_tensor` will be in the following data type and layout:

                .. list-table:: output_tensor
                    :header-rows: 1

                    * - dtype
                      - layout
                    * - BFLOAT16
                      - TILE

            Memory Support:
                - Interleaved: DRAM and L1

            Limitations:
                - All input tensors must be on-device.
                - When :attr:`dim` is not specified (i.e. full product), the :attr:`input_tensor` must be bfloat16, and keepdim=True is not supported  (as this operation results in a scalar).
                - Sharding is not supported for this operation

            Example:
                .. code-block:: python

                    tensor = ttnn.rand((1,2), device=device)
                    output = {1}(tensor, dim=0)
                    output_all_dims = {1}(tensor)

            Example (NC Product):
                .. code-block:: python

                    dims = [0,1]
                    input_shape = [2, 3, 4, 5]
                    output_shape = [1, 1, 4, 5] # shape on any dimension being reduced will be 1

                    input_tensor = ttnn.rand(input_shape, device)
                    output_tensor = ttnn.rand(output_shape, device)

                    output = {1}(input_tensor=input_tensor, output_tensor=output_tensor, dims=dims)
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
