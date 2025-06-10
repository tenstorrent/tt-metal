// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/dropout/dropout.hpp"
#include "ttnn/operations/experimental/dropout/dropout_pybind.hpp"

namespace ttnn::operations::experimental::dropout::detail {
namespace py = pybind11;

void bind_experimental_dropout_operation(py::module& module) {
    auto doc = fmt::format(
        R"doc(

        Applies {0} to :attr:`input_tensor` element-wise.

        .. math::
            \verb|{0}|(\mathrm{{input\_tensor}}_i)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            seed (uint32_t): seed used for RNG.
            probability (float): Dropout probability. In average total_elems * probability elements will be zeroed out.
            scale (float): Scales output tensor. In general scale = 1.0/(1.0-probability).
            memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
            queue_id (int, optional): command queue id. Defaults to `0`.

        Returns:
            ttnn.Tensor: the output tensor.

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
            >>> seed = 124
            >>> prob = 0.2
            >>> output = {1}(tensor,  probability=prob, scale= 1.0/(1.0 - prob), seed=seed)
        )doc",
        ttnn::experimental::dropout.base_name(),
        ttnn::experimental::dropout.python_fully_qualified_name());
    using OperationType = decltype(ttnn::experimental::dropout);
    bind_registered_operation(
        module,
        ttnn::experimental::dropout,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Tensor& input,
               const float probability,
               const float scale,
               const uint32_t seed,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) { return self(input, probability, scale, seed); },
            py::arg("input_tensor"),
            py::arg("probability"),
            py::arg("scale"),
            py::arg("seed"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output_tensor") = std::nullopt});
}
}  // namespace ttnn::operations::experimental::dropout::detail
