// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "ttnn/operations/experimental/dropout/dropout.hpp"

namespace ttnn::operations::experimental::dropout::detail {

void bind_experimental_dropout_operation(nb::module_& mod) {
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
        mod,
        ttnn::experimental::dropout,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const Tensor& input,
               const float probability,
               const float scale,
               const uint32_t seed,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<Tensor>& output_tensor) {
                return self(input, probability, scale, seed, true, memory_config, output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("probability"),
            nb::arg("scale"),
            nb::arg("seed"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}
}  // namespace ttnn::operations::experimental::dropout::detail
