// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "topk_router_gpt.hpp"

namespace ttnn::operations::experimental::topk_router_gpt {

// Free-function wrapper around the registered operation
std::tuple<ttnn::Tensor, ttnn::Tensor> topk_router_gpt_func(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& bias_tensor,
    uint32_t k,
    uint32_t num_experts) {
    return ttnn::experimental::topk_router_gpt(input_tensor, weight_tensor, bias_tensor, k, num_experts);
}

}  // namespace ttnn::operations::experimental::topk_router_gpt

namespace ttnn::operations::experimental::topk_router_gpt::detail {

void bind_topk_router_gpt(nb::module_& mod) {
    ttnn::bind_function<"topk_router_gpt", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused multi-core matmul for GPT-OSS MoE router.

        Parallelizes the router linear layer ([B, hidden] x [hidden, num_experts])
        across 12 DRAM-aligned cores for maximum DRAM read bandwidth.

        Args:
            input_tensor: [B, hidden_dim] bf16 input hidden states
            weight_tensor: [hidden_dim, num_experts] bf16 router weight in DRAM
            bias_tensor: [B, num_experts] bf16 router bias in DRAM, pre-broadcast across batch
            k: Number of top experts (metadata)
            num_experts: Total number of experts
        )doc",
        &ttnn::operations::experimental::topk_router_gpt::topk_router_gpt_func,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("weight_tensor"),
        nb::arg("bias_tensor"),
        nb::arg("k") = 4,
        nb::arg("num_experts") = 128);
}

}  // namespace ttnn::operations::experimental::topk_router_gpt::detail

namespace ttnn::operations::experimental::detail {

void bind_topk_router_gpt(::nanobind::module_& mod) { topk_router_gpt::detail::bind_topk_router_gpt(mod); }

}  // namespace ttnn::operations::experimental::detail
