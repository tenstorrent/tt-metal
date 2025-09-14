// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward_pybind.hpp"

#include <fmt/format.h>

namespace ttnn::operations::experimental::gelu_backward::detail {
namespace py = pybind11;

void bind_experimental_gelu_backward_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Applies the backward pass of the GELU function using ttnn experimental kernels.

        Args:
            grad_tensor (ttnn.Tensor): The input gradient tensor.
            input_tensor (ttnn.Tensor): The input tensor.

        Keyword args:
            approximate (str, optional): "tanh" or "none" (default). The gelu approximation algorithm to use.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for this operation. Defaults to None.
            input_grad (ttnn.Tensor, optional): Preallocated output tensor. Defaults to None.
            queue_id (int, optional): Command queue ID. Defaults to 0.

        Returns:
            ttnn.Tensor: The output tensor.

        Note:
            Supported dtypes, layouts, and ranks:

            .. list-table::
                :header-rows: 1

                * - Dtypes
                    - Layouts
                    - Ranks
                * - BFLOAT16
                    - TILE
                    - 2, 3, 4


        Example:

            >>> grad_tensor = ttnn.from_torch(
            ...     torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
            ...     layout=ttnn.TILE_LAYOUT, device=device
            ... )
            >>> input_tensor = ttnn.from_torch(
            ...     torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True),
            ...     layout=ttnn.TILE_LAYOUT, device=device
            ... )
            >>> output = ttnn.experimental.gelu_bw(grad_tensor, input_tensor)
        )doc");

    using OperationType = decltype(ttnn::experimental::gelu_bw);
    bind_registered_operation(
        module,
        ttnn::experimental::gelu_bw,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& grad_output_tensor,
               const Tensor& input_tensor,
               const std::string& approximate,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor>& input_grad_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, grad_output_tensor, input_tensor, approximate, memory_config, input_grad_tensor);
            },
            py::arg("grad_output_tensor"),
            py::arg("input_tensor"),
            py::kw_only(),
            py::arg("approximate") = "none",
            py::arg("memory_config") = std::nullopt,
            py::arg("input_grad") = std::nullopt,
            py::arg("queue_id") = ttnn::DefaultQueueId});
}
}  // namespace ttnn::operations::experimental::gelu_backward::detail
