// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "moe.hpp"

namespace ttnn::operations::reduction::detail {

void bind_reduction_moe_operation(py::module& module) {
    auto doc =
        R"doc(
        Returns the weight of the zero-th MoE (Mixture of Experts) expert.

        This operation computes expert routing weights for Mixture of Experts models.
        It applies masking and topk selection to determine expert assignments.

        The operation is equivalent to the following PyTorch code:

        .. code-block:: python

            val, ind = torch.topk(input_tensor + expert_mask_tensor, k)
            return torch.sum(torch.softmax(val+topk_mask_tensor, dim=-1)*(ind==0), dim=-1)

        Args:
            input_tensor (ttnn.Tensor): Input tensor for MoE routing. Must be BFLOAT16 with TILE layout.
            expert_mask_tensor (ttnn.Tensor): Expert mask tensor to mask out invalid experts. Must be BFLOAT16 with TILE layout.
            topk_mask_tensor (ttnn.Tensor): Topk mask tensor to mask out everything except topk. Must be BFLOAT16 with TILE layout.
            k (int): Number of top elements to look for. Must be exactly 32.

        Keyword Args:
            memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for the output. Defaults to None.
            output_tensor (Optional[ttnn.Tensor]): Preallocated output tensor. Defaults to None.

        Returns:
            ttnn.Tensor: Output tensor with the same data type and layout as input.

        Note:
            Supported data types and layouts:

            .. list-table::
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - TILE

        Limitations:
            - All tensors must be 4D.
            - For input_tensor: N*C*H must be a multiple of 32. W must be a power of two and ≥64.
            - k must be exactly 32.
            - For topk_mask_tensor: H must be 32 and W must match k (i.e., 32).
            - For expert_mask_tensor: H must be 32 and W must match W of input_tensor.
            - All shape validations are performed on padded shapes.

        Example:

            >>> N, C, H, W = 1, 1, 32, 64
            >>> k = 32
            >>> input_tensor = ttnn.rand([N, C, H, W], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> expert_mask = ttnn.zeros([N, C, 1, W], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> topE_mask = ttnn.zeros([N, C, 1, k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            >>> output = ttnn.moe(input_tensor, expert_mask, topE_mask, k)
        )doc";

    using OperationType = decltype(ttnn::moe);
    bind_registered_operation(
        module,
        ttnn::moe,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_mask_tensor,
               const ttnn::Tensor& topk_mask_tensor,
               const uint16_t k,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor> optional_output_tensor) {
                return self(
                    input_tensor, expert_mask_tensor, topk_mask_tensor, k, memory_config, optional_output_tensor);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("expert_mask_tensor").noconvert(),
            py::arg("topk_mask_tensor").noconvert(),
            py::arg("k") = 32,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::detail
