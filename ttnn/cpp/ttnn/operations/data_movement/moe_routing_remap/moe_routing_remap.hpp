// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteMoeRoutingRemap {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& routing_weights_tensor,
        uint32_t non_zero_weight_size,
        uint32_t expert_parallel_size,
        uint32_t cluster_axis,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto moe_routing_remap =
    ttnn::register_operation<"ttnn::moe_routing_remap", ttnn::operations::data_movement::ExecuteMoeRoutingRemap>();

}  // namespace ttnn
