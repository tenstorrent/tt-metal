// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_grouped_topk.hpp"
#include "device/moe_grouped_topk_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

std::array<Tensor, 2> moe_grouped_topk(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::prim::moe_grouped_topk(
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk
