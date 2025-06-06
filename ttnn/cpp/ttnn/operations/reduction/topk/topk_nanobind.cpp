// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include "ttnn/operations/reduction/topk/topk.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::reduction::detail {

void bind_reduction_topk_operation(nb::module_& mod) {
    auto doc =
        R"doc(topk(input_tensor: ttnn.Tensor, k: int, dim: int, largest: bool, sorted: bool, out : Optional[ttnn.Tensor] = std::nullopt, memory_config: MemoryConfig = std::nullopt, queue_id : [int] = 0) -> Tuple[ttnn.Tensor, ttnn.Tensor]

            Returns the ``k`` largest or ``k`` smallest elements of the given input tensor along a given dimension.

            If ``dim`` is not provided, the last dimension of the input tensor is used.

            If ``largest`` is True, the k largest elements are returned. Otherwise, the k smallest elements are returned.

            The boolean option ``sorted`` if True, will make sure that the returned k elements are sorted.

            Input tensor must have BFLOAT8 or BFLOAT16 data type and TILE_LAYOUT layout.

            Output value tensor will have the same data type as input tensor and output index tensor will have UINT16 data type.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=sorted, *, out=None)

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                k (number): the number of top elements to look for.
                dim (number): the dimension to reduce.
                largest (bool): whether to return the largest or the smallest elements. Defaults to `False`.
                sorted (bool): whether to return the elements in sorted order. Defaults to `False`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                List of ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::topk);
    bind_registered_operation(
        mod,
        ttnn::topk,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t k,
               const int8_t dim,
               const bool largest,
               const bool sorted,
               std::optional<std::tuple<ttnn::Tensor, ttnn::Tensor>> optional_output_tensors,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, input_tensor, k, dim, largest, sorted, memory_config, optional_output_tensors);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("k") = 32,
            nb::arg("dim") = -1,
            nb::arg("largest") = true,
            nb::arg("sorted") = true,
            nb::kw_only(),
            nb::arg("out") = std::nullopt,
            nb::arg("memory_config") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::reduction::detail
