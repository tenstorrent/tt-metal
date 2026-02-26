// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_fused_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "moe_gpt_fused.hpp"

namespace ttnn::operations::experimental::moe_gpt_fused::detail {

void bind_moe_gpt_fused(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::moe_gpt_fused,
        R"doc(
        Experimental fused MoE operation for GPT-OSS with gather + compute + combine.

        Performs sparse token gathering, tilization, SwiGLU(input @ W0, input @ W1) @ W2
        computation via ring all-to-all, and writes results to combine cores.

        Args:
            input_tensor: Input tensor [total_tokens, H] in DRAM (TILE format)
            expert_indices: Expert routing indices [total_tokens, K] uint16
            expert_scores: Expert routing scores [total_tokens, K] bfloat16
            w0_w1_tensor: Interleaved W0/W1 weight tensor (Bfp4_b, DRAM-sharded)
            w2_tensor: W2 weight tensor (Bfp4_b, DRAM-sharded)
            num_experts: Total number of experts
            layer_id: Layer index for weight offset calculation
            experts_per_device: Number of experts assigned to this device
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("expert_indices"),
            nb::arg("expert_scores"),
            nb::arg("w0_w1_tensor"),
            nb::arg("w2_tensor"),
            nb::arg("num_experts"),
            nb::arg("layer_id"),
            nb::arg("experts_per_device") = 4,
        });
}

}  // namespace ttnn::operations::experimental::moe_gpt_fused::detail
