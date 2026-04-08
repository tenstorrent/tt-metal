// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor moe_routing_remap(
    const Tensor& routing_weights_tensor,
    uint32_t non_zero_weight_size,
    uint32_t expert_parallel_size,
    uint32_t cluster_axis,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
