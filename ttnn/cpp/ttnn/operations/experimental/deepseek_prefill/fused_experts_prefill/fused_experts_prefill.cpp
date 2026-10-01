// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_prefill.hpp"

#include "device/fused_experts_prefill_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill {

ttnn::Tensor fused_experts_prefill(
    const ttnn::Tensor& x_tok,
    const ttnn::Tensor& routing_scores,
    const std::vector<ttnn::Tensor>& gate_up_weights,
    const std::vector<ttnn::Tensor>& down_weights,
    uint32_t intermediate_size,
    float swiglu_limit,
    uint32_t top_k,
    float routed_scaling_factor,
    float routing_eps,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& routing_indices,
    const std::optional<ttnn::Tensor>& ranking_scores) {
    // [1, top_k, T, H] ROW_MAJOR: per (slot, token) weighted expert outputs.
    ttnn::Tensor partials = ttnn::prim::fused_experts_prefill(
        x_tok,
        routing_scores,
        gate_up_weights,
        down_weights,
        intermediate_size,
        swiglu_limit,
        top_k,
        routed_scaling_factor,
        routing_eps,
        std::nullopt,
        routing_indices,
        ranking_scores);

    // Weighted sum over the top_k slots.
    ttnn::Tensor tiled = ttnn::to_layout(partials, tt::tt_metal::Layout::TILE);
    return ttnn::sum(tiled, /*dim_arg=*/1, /*keepdim=*/true, memory_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill
