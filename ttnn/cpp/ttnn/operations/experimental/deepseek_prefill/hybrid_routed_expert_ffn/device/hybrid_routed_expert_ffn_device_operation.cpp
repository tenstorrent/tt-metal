// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn_device_operation.hpp"

#include "hybrid_program_factory.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"

#include <set>

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

namespace {

void validate_overlap(const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t) {
    TT_FATAL(
        t.dispatched_metadata.has_value() && t.expert_offsets.has_value() &&
            t.replicated_global_expert_idx_table.has_value() && t.expert_region_offsets.has_value(),
        "hybrid routed expert overlapped with combine needs dispatched_metadata, expert_offsets, the replicated "
        "global_expert_idx_table and expert_region_offsets");
    TT_FATAL(
        op.fwd_arrived.has_value() && op.final_arrived.has_value() && op.expert_go.has_value(),
        "hybrid routed expert overlapped with combine needs the caller's fwd_arrived, final_arrived and expert_go "
        "global semaphores, created once and passed on every call");
    TT_FATAL(
        op.fwd_arrived->device() == t.x.device() && op.final_arrived->device() == t.x.device() &&
            op.expert_go->device() == t.x.device(),
        "hybrid routed expert overlapped with combine: fwd_arrived, final_arrived and expert_go must belong to the "
        "same mesh device as dispatched_buffer");
    TT_FATAL(
        op.fwd_arrived->address() != op.final_arrived->address() &&
            op.fwd_arrived->address() != op.expert_go->address() &&
            op.final_arrived->address() != op.expert_go->address(),
        "hybrid routed expert overlapped with combine: fwd_arrived (0x{:x}), final_arrived (0x{:x}) and expert_go "
        "(0x{:x}) count independently, so they must be three distinct semaphores",
        op.fwd_arrived->address(),
        op.final_arrived->address(),
        op.expert_go->address());
    // Combine bakes these addresses into its compile-time arguments, so the hashed copies must be the live ones.
    TT_FATAL(
        op.fwd_arrived_addr == op.fwd_arrived->address() && op.final_arrived_addr == op.final_arrived->address() &&
            op.expert_go_addr == op.expert_go->address(),
        "hybrid routed expert overlapped with combine: the hashed semaphore addresses do not match the semaphores");
    const uint32_t channels_needed = combine::channel_semaphore_count(
        static_cast<uint32_t>(t.x.device()->shape()[static_cast<int32_t>(op.combine_axis)]));
    TT_FATAL(
        op.channel_semaphores.size() == channels_needed,
        "hybrid routed expert overlapped with combine: {} channel semaphores passed, combine needs {} on this "
        "mesh -- see hybrid_routed_expert_moe_channel_semaphore_count",
        op.channel_semaphores.size(),
        channels_needed);
    TT_FATAL(
        op.channel_semaphore_addrs.size() == op.channel_semaphores.size(),
        "hybrid routed expert overlapped with combine: {} hashed channel addresses for {} semaphores",
        op.channel_semaphore_addrs.size(),
        op.channel_semaphores.size());
    // Each counts something different, so a repeat is one counter serving two purposes: not an error anywhere,
    // just wrong numbers. The trio above is already checked pairwise; this folds the channels in with it.
    std::set<uint32_t> distinct{op.fwd_arrived_addr, op.final_arrived_addr, op.expert_go_addr};
    for (uint32_t i = 0; i < channels_needed; i++) {
        TT_FATAL(
            op.channel_semaphores[i].device() == t.x.device(),
            "hybrid routed expert overlapped with combine: channel semaphore {} belongs to another mesh device",
            i);
        TT_FATAL(
            op.channel_semaphore_addrs[i] == op.channel_semaphores[i].address(),
            "hybrid routed expert overlapped with combine: hashed address of channel semaphore {} does not match "
            "the semaphore",
            i);
        TT_FATAL(
            distinct.insert(op.channel_semaphore_addrs[i]).second,
            "hybrid routed expert overlapped with combine: channel semaphore {} repeats an address (0x{:x}) "
            "already in use",
            i,
            op.channel_semaphore_addrs[i]);
    }
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
