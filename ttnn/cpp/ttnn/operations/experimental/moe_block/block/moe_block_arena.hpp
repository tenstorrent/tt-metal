// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The block's one L1 arena a call: the fused head runs in it, then the routed experts + combine (both take arena=).

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/moe_block/routed_experts_combine/device/routed_experts_combine_device_operation_types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn::operations::experimental::moe_block {

class MoEBlockResources;
struct MoEDtypes;

// The spec of the arena the block allocates once a call (ttnn.allocate_tensor_on_device), passes to
// fused_head(..., arena=) and routed_experts_combine(..., arena=) and frees after the combine: the combine's arena
// (every worker core, height-sharded L1) at max(the combine's need, the head's arena bytes) a core. Plans the head
// (dtypes, accumulators, l1_free as fused_head's) and the combine (the remaining arguments as routed_experts_combine's;
// combine.n_acc, when set, must be accumulators) on first use. rows_cap is the head's hand-off table's.
tt::tt_metal::TensorSpec moe_block_arena_spec(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout = {},
    const routed_experts_combine::CombineConfig& combine = {},
    std::optional<int64_t> l1_free = std::nullopt);

}  // namespace ttnn::operations::experimental::moe_block
