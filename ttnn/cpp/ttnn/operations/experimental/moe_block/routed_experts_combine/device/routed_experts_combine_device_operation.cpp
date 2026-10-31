// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_experts_combine_device_operation.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::routed_experts_combine {

void RoutedExpertsCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& p, const tensor_args_t& t) {
    TT_FATAL(p.plan != nullptr, "routed_experts_combine: no combine plan");
    // SEAM(flat): the flat builder is not integrated yet - without its y writers no row reaches the combine
    TT_FATAL(!p.flat.writers.empty(), "routed_experts_combine: no y writers (the flat routed expert's builder)");
    TT_FATAL(!p.dispatch_in_program, "routed_experts_combine: the in-program dispatch phase is not integrated yet");
    TT_FATAL(
        t.plan.dtype() == DataType::UINT32 && t.plan.logical_volume() >= kPlanWords, "routed_experts_combine: plan");
    TT_FATAL(t.table.dtype() == DataType::UINT32, "routed_experts_combine: table");
    TT_FATAL(
        t.out.dtype() == DataType::BFLOAT16 && t.out.layout() == Layout::ROW_MAJOR &&
            t.out.logical_shape()[-1] == p.W && t.out.logical_shape()[-2] == p.S,
        "routed_experts_combine: out [.., S, Hr] bf16 row-major");
    TT_FATAL(t.lut.has_value() == p.plan->perchip, "routed_experts_combine: the accumulator LUT iff per-chip roles");
}

ttsl::hash::hash_t RoutedExpertsCombineDeviceOperation::compute_program_hash(
    const operation_attributes_t& p, const tensor_args_t& t) {
    return ttsl::hash::hash_objects_with_default_seed(
        p.combine,
        p.S,
        p.W,
        p.dispatch_in_program,
        p.cells,
        p.flat.writers,
        p.flat.words_offset,
        p.plan_key,
        t.plan.tensor_spec(),
        t.table.tensor_spec(),
        t.out.tensor_spec(),
        t.bar.tensor_spec(),
        t.hcache.tensor_spec(),
        t.lut.has_value());
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
