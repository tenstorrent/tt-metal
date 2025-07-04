// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "sort_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/decorators.hpp"
#include "sort.hpp"

namespace ttnn::operations::experimental::reduction::sort::detail {

void bind_reduction_sort_operation(nb::module_& mod) {
    auto doc =
        R"doc(
            Sorts the elements of the input tensor along the specified dimension in ascending order by default.
            If no dimension is specified, the last dimension of the input tensor is used.

            This operation is functionally equivalent to the following PyTorch code:

            .. code-block:: python

                return torch.sort(input_tensor, dim=-1)

            Parameters:
                * `input_tensor` (Tensor): The input tensor to be sorted.

            Keyword Arguments:
                * `dim` (int, optional): The dimension along which to sort. Defaults to -1 (last dimension).
                * `descending` (bool, optional): If `True`, sorts in descending order. Defaults to `False`.
                * `stable` (bool, optional): If `True`, ensures the original order of equal elements is preserved. Defaults to `False`.
                * `memory_config` (MemoryConfig, optional): Specifies the memory configuration for the output tensor. Defaults to `None`.
                * `out` (tuple of Tensors, optional): Preallocated output tensors for the sorted values and indices. Defaults to `None`.

            Additional info:
                * For now the `stable` argument is not supported.

            Example:

            .. code-block:: python

                import ttnn

                # Create a tensor
                input_tensor = ttnn.Tensor([3, 1, 2])

                # Sort the tensor in ascending order
                sorted_tensor, indices = ttnn.experimental.sort(input_tensor)

                # Sort the tensor in descending order
                sorted_tensor_desc, indices_desc = ttnn.experimental.sort(input_tensor, descending=True)

                # Sort along a specific dimension
                input_tensor_2d = ttnn.Tensor([[3, 1, 2], [6, 5, 4]])
                sorted_tensor_dim, indices_dim = ttnn.experimental.sort(input_tensor_2d, dim=1)
        )doc";

    using OperationType = decltype(ttnn::experimental::sort);
    bind_registered_operation(
        mod,
        ttnn::experimental::sort,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const bool descending,
               const bool stable,
               std::optional<std::tuple<ttnn::Tensor&, ttnn::Tensor&>> optional_output_tensors,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) -> std::vector<ttnn::Tensor> {
                return self(queue_id, input_tensor, dim, descending, stable, memory_config, optional_output_tensors);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dim") = -1,
            nb::arg("descending") = false,
            nb::arg("stable") = false,
            nb::kw_only(),
            nb::arg("out") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::reduction::sort::detail
