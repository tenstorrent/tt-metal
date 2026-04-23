// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "post_combine_reduce_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "post_combine_reduce.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce::detail {

void bind_post_combine_reduce(nb::module_& mod) {
    ttnn::bind_function<"post_combine_reduce", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Fused post-combine reduce operation for DeepSeek / GPT-OSS MoE.

            Replaces the inefficient sequence of:
            1. ttnn.to_layout() - ROW_MAJOR -> TILE_LAYOUT with fillpad (8->32 experts)
            2. ttnn.mul() - broadcast weights across embedding dimension
            3. ttnn.sum() - reduce over expert dimension

            With a single fused kernel that eliminates 300% padding overhead.

            Two expert-skip strategies are supported, selected by whether the
            optional ``indices`` and ``expert_dispatch_table`` tensors are
            supplied:

            * Both provided — DeepSeek path. The kernel skips experts whose
              dispatch_table entry is -1 (non-local). Required when upstream
              combine does not zero non-local expert outputs.
            * Both omitted — GPT-OSS path. The kernel skips experts whose
              routing weight is exactly zero. Requires upstream to have
              zeroed non-local routing weights.
            * Supplying only one raises an error.

            Args:
                combine_output (ttnn.Tensor): MoE combine output in ROW_MAJOR layout.
                    Shape: [batch, dispatch_group_size, seq_len, num_experts_per_tok, emb_dim]
                    Example: [1, 1, 3200, 8, 7168]

                weights (ttnn.Tensor): Gate weights for broadcast multiply.
                    Shape: [batch, dispatch_group_size, seq_len, num_experts_per_tok]
                    Example: [1, 1, 3200, 8]

                expert_dim (int, optional): Dimension to reduce over. Defaults to 3.

                indices (ttnn.Tensor, optional): Global expert IDs per token/slot, INT32.
                    Shape: [batch, dispatch_group_size, seq_len, num_experts_per_tok].
                    Required together with expert_dispatch_table for the DeepSeek path.

                expert_dispatch_table (ttnn.Tensor, optional): Dispatch table mapping
                    expert ID to chip ID within dispatch group, INT32. -1 means non-local.
                    Shape: [num_routed_experts] (sharded per dispatch group).
                    Required together with indices for the DeepSeek path.

                output_memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
                    Defaults to L1_MEMORY_CONFIG.

            Returns:
                ttnn.Tensor: Reduced output in TILE_LAYOUT ready for reduce_scatter.
                    Shape: [batch, dispatch_group_size, seq_len, emb_dim]
                    Example: [1, 1, 3200, 7168]
        )doc",
        &ttnn::operations::experimental::deepseek_prefill::post_combine_reduce::post_combine_reduce,
        nb::arg("combine_output").noconvert(),
        nb::arg("weights").noconvert(),
        nb::kw_only(),
        nb::arg("expert_dim") = 3,
        nb::arg("indices") = nb::none(),
        nb::arg("expert_dispatch_table") = nb::none(),
        nb::arg("output_memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_post_combine_reduce(::nanobind::module_& mod) { post_combine_reduce::detail::bind_post_combine_reduce(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
