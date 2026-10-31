// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.routed_experts_combine: the flat routed experts and the sparse combine in one program per
// chip (optionally the dispatch as the combine ports' first phase). See README.md.

#pragma once

#include <cstdint>

#include "device/routed_experts_combine_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block {
class MoEBlockResources;
}

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// dispatched: the gathered rows [N S, Hr] bf16 row-major (the dispatch's output; with experts_during_dispatch its
// buffer, filled in this program from route, the route words [1, 1, 1, RW S]); counts / regions / token_index /
// table / plan: the fused head's planner outputs; global_expert_ids / gate_up_weights / down_weights: the layer's flat
// expert (prepared for the plan of this config). Returns y_own [1, 1, S, Hr] bf16 row-major (persistent: the
// resources' "routed_experts_combine/out").
ttnn::Tensor routed_experts_combine(
    const ttnn::Tensor& dispatched,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& regions,
    const ttnn::Tensor& token_index,
    const ttnn::Tensor& table,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& global_expert_ids,
    const ttnn::Tensor& gate_up_weights,
    const ttnn::Tensor& down_weights,
    MoEBlockResources& resources,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const std::optional<ttnn::Tensor>& route = std::nullopt,
    const flat_routed_expert::FlatLayoutOptions& layout = {},
    const CombineConfig& combine = {});

// The flat expert's config this op builds (for preparing the layer's weights with its plan).
flat_routed_expert::FlatRoutedExpertConfig routed_experts_combine_flat_config(
    MoEBlockResources& resources,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout = {});

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
