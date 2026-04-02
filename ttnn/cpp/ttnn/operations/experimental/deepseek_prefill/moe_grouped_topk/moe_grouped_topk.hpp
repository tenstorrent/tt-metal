// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

std::array<Tensor, 2> moe_grouped_topk(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale = 1.0f,
    float epsilon = 1e-20f,
    bool stable_sort = false,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk
