// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "dropout.hpp"
#include "ttnn/types.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::dropout::detail {

void bind_experimental_dropout_operation(nb::module_& mod) {
    const auto* doc = R"doc(
        Applies dropout to :attr:`input_tensor` element-wise.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            probability (float): Dropout probability. In average total_elems * probability elements will be zeroed out.
            scale (float): Scales output tensor. In general scale = 1.0/(1.0-probability).
            seed (uint32_t): seed used for RNG.

        Keyword Args:
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
            >>> output = ttnn.experimental.dropout(tensor, probability=prob, scale=1.0/(1.0 - prob), seed=seed)
        )doc";

    mod.def(
        "dropout",
        [](const Tensor& input,
           float probability,
           float scale,
           uint32_t seed,
           [[maybe_unused]] const std::optional<MemoryConfig>& memory_config,
           [[maybe_unused]] const std::optional<Tensor>& output_tensor) {
            return ttnn::experimental::dropout(input, probability, scale, seed);
        },
        doc,
        nb::arg("input_tensor"),
        nb::arg("probability"),
        nb::arg("scale"),
        nb::arg("seed"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::experimental::dropout::detail
