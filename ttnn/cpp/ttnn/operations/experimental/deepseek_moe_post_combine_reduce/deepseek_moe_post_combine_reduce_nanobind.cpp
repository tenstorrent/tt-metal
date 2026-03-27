// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe_post_combine_reduce_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_reduce/deepseek_moe_post_combine_reduce.hpp"

namespace ttnn::operations::experimental::detail {

void bind_deepseek_moe_post_combine_reduce(nb::module_& mod) {
    ttnn::bind_function<"deepseek_moe_post_combine_reduce", "ttnn.experimental.">(
        mod,
        R"doc(
            Fused post-combine reduce operation for DeepSeek MoE.

            Replaces the inefficient sequence of:
            1. ttnn.to_layout() - ROW_MAJOR → TILE_LAYOUT with fillpad (8→32 experts)
            2. ttnn.mul() - broadcast weights across embedding dimension
            3. ttnn.sum() - reduce over expert dimension

            With a single fused kernel that eliminates 300% padding overhead.

            Args:
                combine_output (ttnn.Tensor): MoE combine output in ROW_MAJOR layout.
                    Shape: [batch, dispatch_group_size, seq_len, num_experts_per_tok, emb_dim]
                    Example: [1, 1, 256, 8, 7168]

                weights (ttnn.Tensor): Gate weights for broadcast multiply.
                    Shape: [batch, dispatch_group_size, seq_len, num_experts_per_tok]
                    Example: [1, 1, 256, 8]

                expert_dim (int, optional): Dimension to reduce over. Defaults to 3.

                output_memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
                    Defaults to L1_MEMORY_CONFIG.

            Returns:
                ttnn.Tensor: Reduced output in TILE_LAYOUT ready for reduce_scatter.
                    Shape: [batch, dispatch_group_size, seq_len, emb_dim]
                    Example: [1, 1, 256, 7168]

            Example:
                >>> combine_output = ttnn.zeros([1, 1, 256, 8, 7168], layout=ttnn.ROW_MAJOR_LAYOUT)
                >>> weights = ttnn.ones([1, 1, 256, 8])
                >>> result = ttnn.experimental.deepseek_moe_post_combine_reduce(combine_output, weights)
                >>> print(result.shape)
                [1, 1, 256, 7168]
        )doc",
        &ttnn::experimental::deepseek_moe_post_combine_reduce,
        nb::arg("combine_output").noconvert(),
        nb::arg("weights").noconvert(),
        nb::kw_only(),
        nb::arg("expert_dim") = 3,
        nb::arg("output_memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::detail