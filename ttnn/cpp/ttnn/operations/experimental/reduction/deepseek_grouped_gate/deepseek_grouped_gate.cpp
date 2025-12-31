// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_grouped_gate.hpp"
#include "device/deepseek_grouped_gate_device_operation.hpp"

namespace ttnn::operations::experimental::reduction {

std::array<Tensor, 2> DeepseekGroupedGateOperation::invoke(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::prim::deepseek_grouped_gate(
        scores,
        bias,
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        route_scale,
        epsilon,
        output_mem_config);
}

}  // namespace ttnn::operations::experimental::reduction
