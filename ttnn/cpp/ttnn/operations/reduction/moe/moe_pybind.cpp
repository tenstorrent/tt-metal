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
        R"doc(moe(input_tensor: ttnn.Tensor, expert_mask_tensor: ttnn.Tensor, topk_mask_tensor: ttnn.Tensor, k: int, out : Optional[ttnn.Tensor] = std::nullopt, memory_config: MemoryConfig = std::nullopt) -> ttnn.Tensor

            Returns the weight of the zero-th MoE expert.

            Equivalent PyTorch code:

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

            Returns:
                ttnn.Tensor: the output tensor.

            Note:
                The :attr:`input_tensor`, :attr:`expert_mask_tensor`, and :attr:`topk_mask_tensor` must match the following data type and layout:

                    .. list-table::
                        :header-rows: 1

                        * - dtype
                            - layout
                        * - BFLOAT16
                            - TILE

                The output tensor will match the data type and layout of the input tensor.

            Limitations:
                - Tensors must be 4D.
                - For the :attr:`input_tensor`, N*C*H must be a multiple of 32. The last dimension must be a power of two and ≥64.
                - :attr:`k` must be exactly 32.
                - For the :attr:`topk_mask_tensor`, H must be 32 and W must match :attr:`k` (i.e. 32).
                - For the :attr:`expert_mask_tensor`, H must be 32 and W must match W of the :attr:`input_tensor`.
                - All of the shape validations are performed on padded shapes.

            Example:
                N, C, H, W = 1, 1, 32, 64
                k = 32

                input_tensor = ttnn.rand([N, C, H, W], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                expert_mask = ttnn.zeros([N, C, 1, W], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                topE_mask = ttnn.zeros([N, C, 1, k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

                ttnn_output = ttnn.moe(input_tensor, expert_mask, topE_mask, k)

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
