// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "device/moe_routing_remap_device_operation.hpp"
#include "moe_routing_remap.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteMoeRoutingRemap::invoke(
    const ttnn::Tensor& routing_weights_tensor,
    const uint32_t non_zero_weight_size,
    const uint32_t expert_parallel_size,
    const uint32_t cluster_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    return ttnn::prim::moe_routing_remap(
        routing_weights_tensor,
        non_zero_weight_size,
        expert_parallel_size,
        cluster_axis,
        memory_config,
        optional_output_tensor);
}

}  // namespace ttnn::operations::data_movement
