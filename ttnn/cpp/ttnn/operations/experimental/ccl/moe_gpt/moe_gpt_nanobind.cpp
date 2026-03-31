// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "moe_gpt.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::moe_gpt::detail {

namespace {

std::vector<ttnn::Tensor> moe_gpt_wrapper(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices,
    const ttnn::Tensor& expert_scores,
    const ttnn::Tensor& expert_mapping,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    uint32_t output_height_shard_dim,
    uint32_t output_width_shard_dim,
    uint32_t hidden_size,
    std::optional<uint32_t> cluster_axis) {
    return ttnn::experimental::moe_gpt(
        input_tensor,
        expert_indices,
        expert_scores,
        expert_mapping,
        w0_w1_tensor,
        w2_tensor,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
        cluster_axis);
}

}  // namespace

void bind_moe_gpt(nb::module_& mod) {
    const auto* doc =
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
            hidden_size: Hidden dimension size (default: 2880)
            cluster_axis: Optional cluster axis for multi-device dispatch
        )doc";

    ttnn::bind_function<"moe_gpt", "ttnn.experimental.">(
        mod,
        doc,
        moe_gpt_wrapper,
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
        nb::arg("cluster_axis") = nb::none());
}

}  // namespace ttnn::operations::experimental::moe_gpt::detail
