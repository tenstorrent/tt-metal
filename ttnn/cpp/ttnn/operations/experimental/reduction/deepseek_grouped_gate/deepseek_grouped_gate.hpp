// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::reduction {

struct DeepseekGroupedGateOperation {
    static std::array<Tensor, 2> invoke(
        const Tensor& scores,
        const Tensor& bias,
        uint32_t n_groups,
        uint32_t summed_experts_per_group,
        uint32_t topk_groups,
        uint32_t n_activated_experts,
        float route_scale = 1.0f,
        float epsilon = 1e-20f,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::reduction

namespace ttnn::experimental {
constexpr auto deepseek_grouped_gate = ttnn::register_operation<
    "ttnn::experimental::deepseek_grouped_gate",
    ttnn::operations::experimental::reduction::DeepseekGroupedGateOperation>();
}  // namespace ttnn::experimental
