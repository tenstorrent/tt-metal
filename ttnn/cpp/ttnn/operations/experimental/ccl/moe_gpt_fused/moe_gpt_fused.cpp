// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gpt_fused.hpp"
#include "device/moe_gpt_fused_device_operation.hpp"

namespace ttnn::operations::experimental::moe_gpt_fused {

std::vector<ttnn::Tensor> ExecuteMoEGPTFused::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices,
    const ttnn::Tensor& expert_scores,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    uint32_t num_experts,
    uint32_t layer_id,
    uint32_t experts_per_device) {
    return ttnn::prim::moe_gpt_fused(
        input_tensor,
        expert_indices,
        expert_scores,
        w0_w1_tensor,
        w2_tensor,
        num_experts,
        layer_id,
        experts_per_device);
}

}  // namespace ttnn::operations::experimental::moe_gpt_fused
