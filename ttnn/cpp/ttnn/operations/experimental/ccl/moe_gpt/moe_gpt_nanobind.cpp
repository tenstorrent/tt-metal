// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_nanobind.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "moe_gpt.hpp"

namespace ttnn::operations::experimental::moe_gpt::detail {

void bind_moe_gpt(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::moe_gpt,
        R"doc(
        Experimental, high-performance MoE operation for GPT.

        Args:
            input_tensor: Sparse token buffer for tilize phase
            expert_indices: Expert routing indices
            expert_scores: Expert routing scores
            expert_mapping: Expert-to-device mapping
            w0_w1_tensor: Interleaved tensors for first and second matmul
            w2_tensor: Weight tensor for third matmul
            output_height_shard_dim: Height dimension of combine core grid (default: 4)
            output_width_shard_dim: Width dimension of combine core grid (default: 3)
            cluster_axis: Optional cluster axis for multi-device dispatch
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("expert_indices"),
            nb::arg("expert_scores"),
            nb::arg("expert_mapping"),
            nb::arg("w0_w1_tensor"),
            nb::arg("w2_tensor"),
            nb::arg("output_height_shard_dim") = 4,
            nb::arg("output_width_shard_dim") = 3,
            nb::arg("hidden_size") = 2880,
            nb::arg("cluster_axis") = nb::none(),
        });
}

}  // namespace ttnn::operations::experimental::moe_gpt::detail
