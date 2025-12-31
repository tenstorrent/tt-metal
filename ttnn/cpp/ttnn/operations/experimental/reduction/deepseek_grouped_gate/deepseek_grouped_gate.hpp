// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using tt::tt_metal::MemoryConfig;

namespace ttnn::experimental {

std::array<Tensor, 2> deepseek_grouped_gate(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale = 1.0f,
    float epsilon = 1e-20f,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::experimental
