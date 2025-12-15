// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_pybind.hpp"

#include "gather.hpp"

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_gather_operation(py::module& module) {
    const auto* doc = R"doc(
        The `gather` operation extracts values from the input tensor based on indices provided in the index tensor along a specified dimension.

        The input tensor and the index tensor must have the same number of dimensions.
        For all dimensions except the specified one (`dim`), the size of the index tensor must not exceed the size of the input tensor.
        The output tensor will have the same shape as the index tensor. Note that the input and index tensors do not broadcast against each other.

        Args:
            input (ttnn.Tensor): The source tensor from which values are gathered.
            dim (int): The dimension along which values are gathered.
            index (ttnn.Tensor): A tensor containing the indices of elements to gather, with the same number of dimensions as the input tensor.

        Keyword Arguments:
            sparse_grad (bool, optional): If `True`, the gradient computation will be sparse. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            out (ttnn.Tensor, optional): A preallocated tensor to store the gathered values. Defaults to `None`.
            sub_core_grids (ttnn.CoreRangeSet, optional): Custom core range set for operation execution. Allows specification of which cores should be used for the operation. Defaults to `None`.

        Additional Information:
            * Currently, the `sparse_grad` argument is not supported.

        Note:

            Supported dtypes and layout for input tensor values:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16, FLOAT32
                  - TILE
                * - UINT16, UINT32
                  - TILE
                * - INT32
                  - TILE

            Supported dtypes and layout for index tensor values:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - UINT16, UINT32
                  - TILE

        Memory Support:
            - Interleaved: DRAM and L1
    )doc";

    using OperationType = decltype(ttnn::gather);
    bind_registered_operation(
        module,
        ttnn::gather,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const ttnn::Tensor& input_index_tensor,
               const bool sparse_grad,
               std::optional<ttnn::Tensor> optional_output_tensor,
               const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
               const std::optional<CoreRangeSet>& sub_core_grids) -> Tensor {
                return self(
                    input_tensor,
                    dim,
                    input_index_tensor,
                    sparse_grad,
                    memory_config,
                    optional_output_tensor,
                    sub_core_grids);
            },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::arg("index"),
            py::kw_only(),
            py::arg("sparse_grad") = false,
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt});
}

}  // namespace ttnn::operations::data_movement::detail
