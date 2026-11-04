// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.routed_experts_combine: the flat routed experts and the sparse combine in one program per
// chip. See README.md.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "device/combine_program.hpp"
#include "device/routed_experts_combine_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe_block {
class MoEBlockResources;
}

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

// dispatched: the gathered rows [N S, Hr] bf16 row-major (the dispatch's output); counts / regions / token_index /
// table / plan: the fused head's planner outputs; global_expert_ids / gate_up_weights / down_weights: the layer's flat
// expert (prepared for the plan of this config). Returns y_own [1, 1, S, Hr] bf16 row-major (persistent: the
// resources' "routed_experts_combine/out"). arena: the block's shared L1 arena (routed_experts_combine_arena_spec; the
// fused head ran in it first), else the op allocates one for the call.
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
    const std::optional<ttnn::Tensor>& arena = std::nullopt,
    const flat_routed_expert::FlatLayoutOptions& layout = {},
    const CombineConfig& combine = {});

// The combine's L1 plan at a dispatch row group: the flat expert's config fitted into the combine's arena (the
// largest free L1 block less what stays below it, or combine.arena_bytes), the static CBs below the arena and the
// combine's plan. head_port_cb: the fused head's static CBs when it runs in the same arena (the arena stays above
// them). Creates nothing on the device (the block's L1 planner compares groups with it).
struct CombineL1Plan {
    flat_routed_expert::FlatRoutedExpertConfig flat;
    uint32_t static_bytes = 0;  // the flat expert's and the dispatch's static CBs a core (the larger)
    uint32_t arena_bytes = 0;   // the combine's arena a core
    std::vector<tt::tt_metal::CoreCoord> cells;
    std::shared_ptr<const CombinePlan> plan;
};
CombineL1Plan plan_routed_experts_combine_l1(
    MoEBlockResources& resources,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const CombineConfig& combine,
    uint32_t dispatch_group,
    uint32_t head_port_cb = 0);

// L1 a core the block's own programs that run while its arena is allocated (the fused head's ports, the dispatch, the
// flat experts + this op's combine) take for their static circular buffers: what the arena must stay above.
uint32_t routed_experts_combine_static_cb_bytes(
    MoEBlockResources& resources,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout = {});

// The L1 arena spec the block allocates once a call and shares between the fused head and this op (arena=): an L1
// tensor height-sharded over every worker core, max(this op's need, other_bytes) a core (other_bytes: the head's
// arena_bytes). rows_cap: the hand-off table's rows (32 x its pages). Plans the op on first use.
tt::tt_metal::TensorSpec routed_experts_combine_arena_spec(
    MoEBlockResources& resources,
    uint32_t rows_cap,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout = {},
    const CombineConfig& combine = {},
    uint32_t other_bytes = 0);

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine
