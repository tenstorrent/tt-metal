// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_grouped_gate.hpp"
#include "device/deepseek_grouped_gate_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

std::array<Tensor, 2> deepseek_grouped_gate(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    const std::optional<MemoryConfig>& output_mem_config) {
    using OperationType = operations::experimental::reduction::DeepseekGroupedGateDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        route_scale,
        epsilon,
        output_mem_config.value_or(scores.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{scores, bias};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
