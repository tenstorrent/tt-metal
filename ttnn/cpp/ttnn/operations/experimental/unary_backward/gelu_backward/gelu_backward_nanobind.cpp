// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gelu_backward_nanobind.hpp"

#include <optional>
#include <string>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward.hpp"

namespace ttnn::operations::experimental::gelu_backward::detail {

void bind_experimental_gelu_backward_operation(nb::module_& mod) {
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
        mod,
        ttnn::experimental::gelu_bw,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& grad_output_tensor,
               const Tensor& input_tensor,
               const std::string& approximate,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<Tensor>& input_grad_tensor) -> ttnn::Tensor {
                return self(grad_output_tensor, input_tensor, approximate, memory_config, input_grad_tensor);
            },
            nb::arg("grad_output_tensor"),
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("approximate") = "none",
            nb::arg("memory_config") = nb::none(),
            nb::arg("input_grad") = nb::none()});
}
}  // namespace ttnn::operations::experimental::gelu_backward::detail
