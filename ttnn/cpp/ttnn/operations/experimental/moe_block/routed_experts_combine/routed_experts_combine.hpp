// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.routed_experts_combine: the flat routed experts and the sparse combine in one program per
// chip. See README.md.

#pragma once

#include <cstdint>

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

// L1 a core the block's own programs that run while its arena is allocated (the dispatch, the flat experts + this
// op's combine) take for their static circular buffers: what the arena must stay above.
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
