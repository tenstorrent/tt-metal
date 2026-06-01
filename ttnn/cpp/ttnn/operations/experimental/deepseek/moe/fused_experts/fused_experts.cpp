// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts.hpp"

#include "device/fused_experts_device_operation.hpp"

namespace ttnn::experimental::deepseek::moe {

Tensor fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::fused_experts(
        input_tensor,
        routing_weights,
        gate_up_weights,
        down_weights,
        num_experts,
        intermediate_size,
        swiglu_limit,
        memory_config);
}

}  // namespace ttnn::experimental::deepseek::moe
