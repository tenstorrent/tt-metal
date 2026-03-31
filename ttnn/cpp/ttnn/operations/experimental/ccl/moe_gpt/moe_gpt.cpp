// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt.hpp"
#include "device/moe_gpt_device_operation.hpp"

namespace ttnn::operations::experimental::moe_gpt {

std::vector<ttnn::Tensor> ExecuteMoEGPT::invoke(
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
    return ttnn::prim::moe_gpt(
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

}  // namespace ttnn::operations::experimental::moe_gpt
