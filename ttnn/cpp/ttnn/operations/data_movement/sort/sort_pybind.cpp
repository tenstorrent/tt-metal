// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_pybind.hpp"

#include "sort.hpp"

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_sort_operation(py::module& module) {
    const auto* doc =
        R"doc(
        Sorts the elements of the input tensor along the specified dimension in ascending order by default.
        If no dimension is specified, the last dimension of the input tensor is used.

        Args:
            input_tensor (ttnn.Tensor): The input tensor to be sorted.

        Keyword Arguments:
            dim (int, optional): The dimension along which to sort. Defaults to `-1` (last dimension).
            descending (bool, optional): If `True`, sorts in descending order. Defaults to `False`.
            stable (bool, optional): If `True`, ensures the original order of equal elements is preserved. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
            out (tuple of ttnn.Tensor, optional): Preallocated output tensors for the sorted values and indices. Defaults to `None`. The index tensor must be of type uint16 or uint32.

        Returns:
            List of ttnn.Tensor: A list containing two tensors: The first tensor contains the sorted values, the second tensor contains the indices of the original elements in the sorted order.

        Additional info:
            * For now the `stable` argument is not supported.

        Note:

            Supported dtypes and layout for input tensor values:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                  - Layouts
                * - BFLOAT16
                  - TILE
                * - UINT16
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

    using OperationType = decltype(ttnn::sort);
    bind_registered_operation(
        module,
        ttnn::sort,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const bool descending,
               const bool stable,
               std::optional<std::tuple<ttnn::Tensor&, ttnn::Tensor&>> optional_output_tensors,
               const std::optional<ttnn::MemoryConfig>& memory_config) -> std::vector<ttnn::Tensor> {
                return self(input_tensor, dim, descending, stable, memory_config, optional_output_tensors);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim") = -1,
            py::arg("descending") = false,
            py::arg("stable") = false,
            py::kw_only(),
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
        });
}

}  // namespace ttnn::operations::data_movement::detail
