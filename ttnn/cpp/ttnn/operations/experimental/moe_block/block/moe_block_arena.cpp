// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_arena.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/moe_block/common/moe_block_config.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"
#include "ttnn/operations/experimental/moe_block/fused_head/fused_head.hpp"
#include "ttnn/operations/experimental/moe_block/routed_experts_combine/routed_experts_combine.hpp"

namespace ttnn::operations::experimental::moe_block {

tt::tt_metal::TensorSpec moe_block_arena_spec(
    MoEBlockResources& resources,
    const MoEDtypes& dtypes,
    uint32_t accumulators,
    uint32_t max_tokens_per_expert,
    uint32_t activation,
    const flat_routed_expert::FlatLayoutOptions& layout,
    const routed_experts_combine::CombineConfig& combine,
    std::optional<int64_t> l1_free) {
    TT_FATAL(
        !combine.n_acc.has_value() || *combine.n_acc == accumulators,
        "moe_block: the combine's accumulators ({}) must be the head's ({})",
        *combine.n_acc,
        accumulators);
    const auto& head = fused_head::setup_fused_head(resources, dtypes, accumulators, l1_free);
    const uint32_t head_bytes = fused_head::head_tensor_shapes(*head.model).arena_bytes;
    const uint32_t rows_cap = head.inputs.table.logical_shape()[-2] * 32;
    auto spec = routed_experts_combine::routed_experts_combine_arena_spec(
        resources, rows_cap, max_tokens_per_expert, activation, layout, combine, head_bytes);
    // (fused_head's own checks on the arena: L1, its cores, its bytes a core)
    const auto& shard = *spec.memory_config().shard_spec();
    TT_FATAL(
        shard.grid.contains(fused_head::head_arena_cores(*head.model)),
        "moe_block: the combine's arena cores do not cover the fused head's");
    TT_FATAL(
        uint64_t(shard.shape[0]) * shard.shape[1] * 2 >= head_bytes,
        "moe_block: the arena holds {} B a core, the fused head needs {} B",
        uint64_t(shard.shape[0]) * shard.shape[1] * 2,
        head_bytes);
    return spec;
}

}  // namespace ttnn::operations::experimental::moe_block
