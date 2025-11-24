// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_gate.hpp"
#include "device/grouped_gate_device_operation.hpp"

namespace ttnn::operations::reduction {

std::tuple<Tensor, Tensor> GroupedGateOperation::invoke(
    const Tensor& scores,
    const Tensor& bias,
    const float route_scale,
    const float epsilon,
    const uint32_t n_groups,
    const uint32_t topk,
    const uint32_t topk_groups,
    const uint32_t n_activated_experts,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::prim::grouped_gate(
        scores, bias, route_scale, epsilon, n_groups, topk, topk_groups, n_activated_experts, output_mem_config);
}

}  // namespace ttnn::operations::reduction
