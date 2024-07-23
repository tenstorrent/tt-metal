// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "moe.hpp"
#include <optional>

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;

void bind_reduction_moe_operation(py::module& module) {
    auto doc =
        R"doc(moe(input_tensor: ttnn.Tensor, topk_mask_tensor: ttnn.Tensor, expert_mask_tensor: ttnn.Tensor, k: int, dim: int, largest: bool, sorted: bool, out : Optional[ttnn.Tensor] = std::nullopt, memory_config: MemoryConfig = std::nullopt, queue_id : [int] = 0) -> ttnn.Tensor

            Returns the ``k`` largest or ``k`` smallest elements of the given input tensor along a given dimension.

            If ``dim`` is not provided, the last dimension of the input tensor is used.

            If ``largest`` is True, the k largest elements are returned. Otherwise, the k smallest elements are returned.

            The boolean option ``sorted`` if True, will make sure that the returned k elements are sorted.

            Input tensor must have BFLOAT8 or BFLOAT16 data type and TILE_LAYOUT layout.

            Output value tensor will have the same data type as input tensor and output index tensor will have UINT16 data type.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.moe(input_tensor, k, dim=dim, largest=largest, sorted=sorted, *, out=None)

            Args:
                * :attr:`input_tensor`: Input Tensor for moe.
                * :attr:`k`: the number of top elements to look for
                * :attr:`dim`: the dimension to reduce
                * :attr:`largest`: whether to return the largest or the smallest elements
                * :attr:`sorted`: whether to return the elements in sorted order

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensors
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensors
                * :attr:`queue_id` (Optional[uint8]): command queue id
        )doc";

    using OperationType = decltype(ttnn::moe);
    bind_registered_operation(
        module,
        ttnn::moe,
        doc,
        ttnn::pybind_overload_t{
            [] (const OperationType& self,
                const ttnn::Tensor& input_tensor,
                const ttnn::Tensor& topk_mask_tensor,
                const ttnn::Tensor& expert_mask_tensor,
                const uint16_t k,
                const std::optional<ttnn::MemoryConfig>& memory_config,
                std::optional<ttnn::Tensor> optional_output_tensor,
                uint8_t queue_id) {
                    return self(queue_id, input_tensor, topk_mask_tensor, expert_mask_tensor, k,
                    memory_config, optional_output_tensor);
                },
                py::arg("input_tensor").noconvert(),
                py::arg("topk_mask_tensor").noconvert(),
                py::arg("expert_mask_tensor").noconvert(),
                py::arg("k") = 32,
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("output_tensor") = std::nullopt,
                py::arg("queue_id") = 0});
}


}  // namespace ttnn::operations::reduction::detail
