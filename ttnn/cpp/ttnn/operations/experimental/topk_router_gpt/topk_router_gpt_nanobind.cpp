// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_nanobind.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "topk_router_gpt.hpp"

namespace ttnn::operations::experimental::topk_router_gpt::detail {

void bind_topk_router_gpt(nb::module_& mod) {
    bind_registered_operation(
        mod,
        ttnn::experimental::topk_router_gpt,
        R"doc(
        Fused multi-core matmul for GPT-OSS MoE router.

        Parallelizes the router linear layer ([B, hidden] x [hidden, num_experts])
        across 12 DRAM-aligned cores for maximum DRAM read bandwidth.

        Args:
            input_tensor: [B, hidden_dim] bf16 input hidden states
            weight_tensor: [hidden_dim, num_experts] bf16 router weight in DRAM
            bias_tensor: [1, num_experts] bf16 router bias in DRAM
            k: Number of top experts (metadata)
            num_experts: Total number of experts
        )doc",
        ttnn::nanobind_arguments_t{
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("weight_tensor"),
            nb::arg("bias_tensor"),
            nb::arg("k") = 4,
            nb::arg("num_experts") = 128,
            nb::arg("untilize_output") = false,
        });
}

}  // namespace ttnn::operations::experimental::topk_router_gpt::detail
