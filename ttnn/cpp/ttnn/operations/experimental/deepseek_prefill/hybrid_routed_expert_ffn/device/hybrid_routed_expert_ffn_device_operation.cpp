// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn_device_operation.hpp"

#include <array>

#include "hybrid_program_factory.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

namespace {

void validate_overlap(const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t) {
    TT_FATAL(
        t.dispatched_metadata.has_value() && t.expert_offsets.has_value() &&
            t.replicated_global_expert_idx_table.has_value() && t.expert_region_offsets.has_value(),
        "hybrid routed expert overlapped with combine needs dispatched_metadata, expert_offsets, the replicated "
        "global_expert_idx_table and expert_region_offsets");
    TT_FATAL(
        op.fwd_arrived.has_value() && op.final_arrived.has_value() && op.expert_go.has_value() &&
            op.launch_credit.has_value(),
        "hybrid routed expert overlapped with combine needs the caller's fwd_arrived, final_arrived, expert_go and "
        "launch_credit global semaphores, created once and passed on every call");
    const std::array<const GlobalSemaphore*, 4> sems = {
        &*op.fwd_arrived, &*op.final_arrived, &*op.expert_go, &*op.launch_credit};
    for (const auto* sem : sems) {
        TT_FATAL(
            sem->device() == t.x.device(),
            "hybrid routed expert overlapped with combine: fwd_arrived, final_arrived, expert_go and launch_credit "
            "must belong to the same mesh device as dispatched_buffer");
    }
    for (size_t i = 0; i < sems.size(); i++) {
        for (size_t j = i + 1; j < sems.size(); j++) {
            TT_FATAL(
                sems[i]->address() != sems[j]->address(),
                "hybrid routed expert overlapped with combine: fwd_arrived (0x{:x}), final_arrived (0x{:x}), "
                "expert_go (0x{:x}) and launch_credit (0x{:x}) count independently, so they must be four distinct "
                "semaphores",
                op.fwd_arrived->address(),
                op.final_arrived->address(),
                op.expert_go->address(),
                op.launch_credit->address());
        }
    }
    // Combine bakes these addresses into its compile-time arguments, so the hashed copies must be the live ones.
    TT_FATAL(
        op.fwd_arrived_addr == op.fwd_arrived->address() && op.final_arrived_addr == op.final_arrived->address() &&
            op.expert_go_addr == op.expert_go->address() && op.launch_credit_addr == op.launch_credit->address(),
        "hybrid routed expert overlapped with combine: the hashed semaphore addresses do not match the semaphores");
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
    if (t.forwarding_buffer.has_value()) {
        const auto expected = combine::forwarding_buffer_spec(
            combine_attributes(op, t),
            ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::token_size_bytes(combine_inputs(t)));
        TT_FATAL(
            t.forwarding_buffer->tensor_spec() == expected,
            "hybrid routed expert overlapped with combine: forwarding_buffer is {} {} {}, combine needs {} {} {}",
            t.forwarding_buffer->logical_shape(),
            t.forwarding_buffer->dtype(),
            t.forwarding_buffer->memory_config(),
            expected.logical_shape(),
            expected.data_type(),
            expected.memory_config());
    }
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
