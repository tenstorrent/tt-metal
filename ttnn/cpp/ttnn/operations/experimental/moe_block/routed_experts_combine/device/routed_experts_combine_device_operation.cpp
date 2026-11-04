// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_device_operation.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

void RoutedExpertsCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& p, const tensor_args_t& t) {
    TT_FATAL(p.plan != nullptr && p.flat_plan != nullptr, "routed_experts_combine: no plans");
    TT_FATAL(!p.flat.writers.empty(), "routed_experts_combine: no y writers");
    TT_FATAL(
        p.flat_config.cmb_rt > 0 && p.flat_config.y_l1, "routed_experts_combine: the flat config's combine overlap");
    TT_FATAL(
        t.plan.dtype() == DataType::UINT32 && t.plan.logical_volume() >= kPlanWords, "routed_experts_combine: plan");
    TT_FATAL(t.table.dtype() == DataType::UINT32, "routed_experts_combine: table");
    TT_FATAL(
        t.out.dtype() == DataType::BFLOAT16 && t.out.layout() == Layout::ROW_MAJOR &&
            t.out.logical_shape()[-1] == p.W && t.out.logical_shape()[-2] == p.S,
        "routed_experts_combine: out [.., S, Hr] bf16 row-major");
    TT_FATAL(t.lut.has_value() == p.plan->perchip, "routed_experts_combine: the accumulator LUT iff per-chip roles");
    TT_FATAL(
        t.flat.arena.buffer()->aligned_size_per_bank() >= p.plan->arena_bytes + kWriterWordsRows * 64,
        "routed_experts_combine: the arena holds {} B a core, the combine and the writers' words need {} B",
        t.flat.arena.buffer()->aligned_size_per_bank(),
        p.plan->arena_bytes + kWriterWordsRows * 64);
}

ttsl::hash::hash_t RoutedExpertsCombineDeviceOperation::compute_program_hash(
    const operation_attributes_t& p, const tensor_args_t& t) {
    return ttsl::hash::hash_objects_with_default_seed(
        p.combine,
        p.S,
        p.W,
        // (the flat expert's per-call choices: not in plan_key; its plan, cells and y writers follow from them and it)
        p.flat_config.max_tokens,
        p.flat_config.activation,
        p.flat_config.layout,
        p.plan_key,
        t.plan.tensor_spec(),
        t.table.tensor_spec(),
        t.out.tensor_spec(),
        t.bar.tensor_spec(),
        t.hcache.tensor_spec(),
        t.lut.has_value(),
        t.flat.x.tensor_spec(),
        t.flat.counts.tensor_spec(),
        t.flat.regions.tensor_spec(),
        t.flat.global_expert_ids.tensor_spec(),
        t.flat.gate_up_weights.tensor_spec(),
        t.flat.down_weights.tensor_spec(),
        t.flat.done_words.tensor_spec(),
        t.flat.arena.tensor_spec(),
        t.flat.words.tensor_spec(),
        t.flat.output.tensor_spec(),
        t.flat.token_index.has_value() ? std::optional(t.flat.token_index->tensor_spec()) : std::nullopt);
}

}  // namespace ttnn::operations::experimental::moe_block::routed_experts_combine

namespace ttnn::prim {
ttnn::Tensor moe_block_routed_experts_combine(
    const ttnn::operations::experimental::moe_block::routed_experts_combine::RoutedExpertsCombineParams& params,
    const ttnn::operations::experimental::moe_block::routed_experts_combine::RoutedExpertsCombineInputs& inputs) {
    using Op = ttnn::operations::experimental::moe_block::routed_experts_combine::RoutedExpertsCombineDeviceOperation;
    return ttnn::device_operation::launch<Op>(params, inputs);
}
}  // namespace ttnn::prim
