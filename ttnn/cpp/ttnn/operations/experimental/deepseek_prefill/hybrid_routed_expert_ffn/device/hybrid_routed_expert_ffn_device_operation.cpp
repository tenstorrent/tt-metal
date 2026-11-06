// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn_device_operation.hpp"

#include "hybrid_program_factory.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_combine_overlap/device/routed_expert_handoff.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

namespace {

void validate_overlap(const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t) {
    TT_FATAL(
        t.dispatched_metadata.has_value() && t.expert_offsets.has_value() &&
            t.replicated_global_expert_idx_table.has_value() && t.expert_region_offsets.has_value(),
        "hybrid routed expert overlapped with combine needs dispatched_metadata, expert_offsets, the replicated "
        "global_expert_idx_table and expert_region_offsets");
    combine::validate_overlap_semaphores(
        "hybrid_routed_expert_moe",
        t.x.device(),
        op.combine_axis,
        op.fwd_arrived,
        op.final_arrived,
        op.expert_go,
        op.channel_semaphores,
        op.fwd_arrived_addr,
        op.final_arrived_addr,
        op.expert_go_addr,
        op.channel_semaphore_addrs);
    TT_FATAL(
        t.l1_arena.has_value() == (op.hybrid_token_threshold > 0),
        "hybrid routed expert overlapped with combine needs an L1 arena exactly when the fused pass runs "
        "(threshold {}, arena {})",
        op.hybrid_token_threshold,
        t.l1_arena.has_value() ? "present" : "absent");
    TT_FATAL(
        op.l1_arena_addr == (t.l1_arena.has_value() ? t.l1_arena->buffer()->address() : 0),
        "hybrid routed expert overlapped with combine: the hashed L1 arena address 0x{:x} is not the arena's",
        op.l1_arena_addr);
    // Combine's untilizers read the routed expert's output as bfloat8_b tiles, as combine does back to back.
    TT_FATAL(
        t.output.layout() == tt::tt_metal::Layout::TILE && t.output.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "hybrid routed expert overlapped with combine writes a bfloat8_b TILE output, got {} {}",
        t.output.dtype(),
        t.output.layout());
    // The framework resolves cache-hit bindings by buffer, and x and the output are both inputs here.
    TT_FATAL(
        t.output.buffer() != t.x.buffer(),
        "hybrid routed expert overlapped with combine needs an output distinct from dispatched_buffer");
    combine::validate_forwarding_buffer(
        "hybrid_routed_expert_moe", t.forwarding_buffer, combine_attributes(op, t), combine_inputs(t));
    ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dDeviceOperation::
        validate_on_program_cache_miss(combine_attributes(op, t), combine_inputs(t));
}

}  // namespace

HybridRoutedExpertFfnDeviceOperation::program_factory_t HybridRoutedExpertFfnDeviceOperation::select_program_factory(
    const operation_attributes_t& op, const tensor_args_t&) {
    if (op.overlap_combine) {
        return HybridOverlapProgramFactory{};
    }
    return HybridSoloProgramFactory{};
}

void HybridRoutedExpertFfnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& t) {
    validate_arguments(op, t);
    if (op.overlap_combine) {
        validate_overlap(op, t);
    }
}

void HybridRoutedExpertFfnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& op, const tensor_args_t& t) {
    // Tensor specs and compile-time attributes participate in the cache key, but buffer addresses
    // do not. Re-run validation so an address-only hit cannot alias the output onto x, move an
    // argument to another device, or hand a half a buffer the cached program was not built for.
    validate_arguments(op, t);
    if (op.overlap_combine) {
        validate_overlap(op, t);
    }
}

HybridRoutedExpertFfnDeviceOperation::spec_return_value_t HybridRoutedExpertFfnDeviceOperation::compute_output_specs(
    const operation_attributes_t& op, const tensor_args_t& t) {
    if (op.overlap_combine) {
        return ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dDeviceOperation::
            compute_output_specs(combine_attributes(op, t), combine_inputs(t));
    }
    return t.output.tensor_spec();
}

HybridRoutedExpertFfnDeviceOperation::tensor_return_value_t HybridRoutedExpertFfnDeviceOperation::create_output_tensors(
    const operation_attributes_t& op, const tensor_args_t& t) {
    if (op.overlap_combine) {
        return create_device_tensor(compute_output_specs(op, t), t.x.device());
    }
    return t.output;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
