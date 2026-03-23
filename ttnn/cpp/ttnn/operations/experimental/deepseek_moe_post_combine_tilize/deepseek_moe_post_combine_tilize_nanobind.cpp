// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "deepseek_moe_post_combine_tilize_nanobind.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/deepseek_moe_post_combine_tilize.hpp"

#include <ttnn-nanobind/small_vector_caster.hpp>
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize::detail {
namespace nb = nanobind;

void bind_deepseek_moe_post_combine_tilize(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Changes data layout of input tensor to TILE.

        Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            output_memory_config (ttnn.MemoryConfig): Memory configuration for the operation.

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    ttnn::bind_function<"deepseek_moe_post_combine_tilize", "ttnn.experimental.">(
        mod,
        doc,
        &ttnn::experimental::deepseek_moe_post_combine_tilize,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("output_memory_config").noconvert());
}
}  // namespace ttnn::operations::experimental::deepseek_moe_post_combine_tilize::detail
