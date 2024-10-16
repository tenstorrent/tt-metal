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
        R"doc(moe(input_tensor: ttnn.Tensor, expert_mask_tensor: ttnn.Tensor, topk_mask_tensor: ttnn.Tensor, k: int, out : Optional[ttnn.Tensor] = std::nullopt, memory_config: MemoryConfig = std::nullopt, queue_id : [int] = 0) -> ttnn.Tensor

            Returns the weight of the zero-th MoE expert.
            Input tensor must have BBFLOAT16 data type and TILE_LAYOUT layout.
            expert_mask_tensor and topk_mask_tensor must have BFLOAT16 data type and TILE_LAYOUT layout.

            Output value tensor will have the same data type as input tensor and output.

            Equivalent pytorch code:

            .. code-block:: python
                val, ind = torch.topk(input_tensor + expert_mask_tensor, k)
                return torch.sum(torch.softmax(val+topk_mask_tensor, dim=-1)*(ind==0), dim=-1)

            Args:
                * :attr:`input_tensor`: Input Tensor for moe.
                * :attr:`expert_mask_tensor`: Expert Mask Tensor for mask to out invalid experts.
                * :attr:`topk_mask_tensor`: Topk Mask Tensor for mask to out everything except topk.
                * :attr:`k`: the number of top elements to look for

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensors
                * :attr:`output_tensor` (Optional[ttnn.Tensor]): preallocated output tensors
                * :attr:`queue_id` (Optional[uint8]): command queue id
        )doc";

    using OperationType = decltype(ttnn::moe);
    bind_registered_operation(module,
                              ttnn::moe,
                              doc,
                              ttnn::pybind_overload_t{[](const OperationType& self,
                                                         const ttnn::Tensor& input_tensor,
                                                         const ttnn::Tensor& expert_mask_tensor,
                                                         const ttnn::Tensor& topk_mask_tensor,
                                                         const uint16_t k,
                                                         const std::optional<ttnn::MemoryConfig>& memory_config,
                                                         std::optional<ttnn::Tensor> optional_output_tensor,
                                                         uint8_t queue_id) {
                                                          return self(queue_id,
                                                                      input_tensor,
                                                                      expert_mask_tensor,
                                                                      topk_mask_tensor,
                                                                      k,
                                                                      memory_config,
                                                                      optional_output_tensor);
                                                      },
                                                      py::arg("input_tensor").noconvert(),
                                                      py::arg("expert_mask_tensor").noconvert(),
                                                      py::arg("topk_mask_tensor").noconvert(),
                                                      py::arg("k") = 32,
                                                      py::kw_only(),
                                                      py::arg("memory_config") = std::nullopt,
                                                      py::arg("output_tensor") = std::nullopt,
                                                      py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::reduction::detail
