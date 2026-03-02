// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::moe_gpt_fused {

struct ExecuteMoEGPTFused {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_indices,
        const ttnn::Tensor& expert_scores,
        const ttnn::Tensor& w0_w1_tensor,
        const ttnn::Tensor& w2_tensor,
        uint32_t num_experts,
        uint32_t layer_id,
        uint32_t experts_per_device);
};

}  // namespace ttnn::operations::experimental::moe_gpt_fused

namespace ttnn::experimental {
constexpr auto moe_gpt_fused = ttnn::register_operation<
    "ttnn::experimental::moe_gpt_fused",
    ttnn::operations::experimental::moe_gpt_fused::ExecuteMoEGPTFused>();
}  // namespace ttnn::experimental
