// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/clip_grad_norm/clip_grad_norm.hpp"
#include "ttnn/operations/experimental/clip_grad_norm/clip_grad_norm_pybind.hpp"

namespace ttnn::operations::experimental::clip_grad_norm::detail {
namespace py = pybind11;

void bind_experimental_clip_grad_norm_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc(
        Clips gradient norm of :attr:`input_tensor` to :attr:`max_norm`.

        The gradient norm is computed using the p-norm (default L2 norm).
        If the computed norm exceeds :attr:`max_norm`, the gradient is scaled by
        max_norm / (computed_norm + eps).

        .. math::
            norm = (sum_i |input_tensor_i|^p)^(1/p)
            scale = min(1, max_norm / (norm + eps))
            output = input_tensor * scale

        Args:
            input_tensor (ttnn.Tensor): the input gradient tensor.

        Keyword Args:
            max_norm (float): Maximum norm value. Gradients are scaled if norm exceeds this value.
            p (float): The norm order. Default is 2.0 (L2 norm). Use float('inf') for max norm.
            eps (float): Small epsilon value added to norm for numerical stability. Default is 1e-12.
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the clipped gradient tensor.

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
            >>> tensor = ttnn.from_torch(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
            >>> max_norm = 2.0
            >>> output = {0}(tensor, max_norm=max_norm, p=2.0, eps=1e-12)
        )doc",
        ttnn::experimental::clip_grad_norm.python_fully_qualified_name());
    using OperationType = decltype(ttnn::experimental::clip_grad_norm);
    bind_registered_operation(
        module,
        ttnn::experimental::clip_grad_norm,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input,
               const float max_norm,
               const float p,
               const float eps,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) { return self(input, max_norm, p, eps); },
            py::arg("input_tensor"),
            py::arg("max_norm"),
            py::arg("p") = 2.0f,
            py::arg("eps") = 1e-12f,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt});
}
}  // namespace ttnn::operations::experimental::clip_grad_norm::detail
