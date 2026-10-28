// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn_device_operation.hpp"

#include <algorithm>

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
    // A hit has the specs and attributes the miss validated, since they make up the cache key, and the device's
    // arch, grid and allocator base are the same device's. What a hit can still change is which buffers it is handed:
    // re-check only that, not the full validation, whose cost is host time on every eager call.
    const auto* device = t.x.device();
    const auto check = [&](const Tensor& tensor, const char* name) {
        TT_FATAL(
            tensor.is_allocated() && tensor.device() == device,
            "hybrid routed expert: {} must be allocated on dispatched_buffer's device",
            name);
    };
    const auto check_optional = [&](const std::optional<Tensor>& tensor, const char* name) {
        if (tensor.has_value()) {
            check(*tensor, name);
        }
    };
    const auto check_all = [&](const std::vector<Tensor>& tensors, const char* name) {
        for (const auto& tensor : tensors) {
            check(tensor, name);
        }
    };
    TT_FATAL(t.x.is_allocated(), "hybrid routed expert: dispatched_buffer must be allocated");
    check_all(t.gate_projs, "gate_projs");
    check_all(t.up_projs, "up_projs");
    check_all(t.down_projs, "down_projs");
    check(t.counts, "counts");
    check(t.global_expert_idx_table, "global_expert_idx_table");
    check(t.output, "output");
    check_optional(t.expert_region_offsets, "expert_region_offsets");
    check_all(t.gate_biases, "gate_biases");
    check_all(t.up_biases, "up_biases");
    check_all(t.down_biases, "down_biases");
    check_optional(t.l1_arena, "l1_arena");
    check_optional(t.dispatched_metadata, "dispatched_metadata");
    check_optional(t.expert_offsets, "expert_offsets");
    check_optional(t.replicated_global_expert_idx_table, "replicated_global_expert_idx_table");
    check_optional(t.forwarding_buffer, "forwarding_buffer");
    if (!op.overlap_combine) {
        return;
    }
    TT_FATAL(
        t.output.buffer() != t.x.buffer(),
        "hybrid routed expert overlapped with combine needs an output distinct from dispatched_buffer");
    // The semaphores are not tensors, so the key holds their addresses, not them.
    TT_FATAL(
        op.fwd_arrived.has_value() && op.final_arrived.has_value() && op.expert_go.has_value(),
        "hybrid routed expert overlapped with combine needs the caller's fwd_arrived, final_arrived and expert_go "
        "global semaphores, created once and passed on every call");
    TT_FATAL(
        op.fwd_arrived->device() == device && op.final_arrived->device() == device && op.expert_go->device() == device,
        "hybrid routed expert overlapped with combine: fwd_arrived, final_arrived and expert_go must belong to the "
        "same mesh device as dispatched_buffer");
    TT_FATAL(
        op.fwd_arrived_addr == op.fwd_arrived->address() && op.final_arrived_addr == op.final_arrived->address() &&
            op.expert_go_addr == op.expert_go->address(),
        "hybrid routed expert overlapped with combine: the hashed semaphore addresses do not match the semaphores");
    TT_FATAL(
        op.l1_arena_addr == (t.l1_arena.has_value() ? t.l1_arena->buffer()->address() : 0),
        "hybrid routed expert overlapped with combine: the hashed L1 arena address 0x{:x} is not the arena's",
        op.l1_arena_addr);
}

namespace {

// A per-expert list -- the projections or biases of one kind -- normally holds one spec once per expert. When every
// tensor in it is a device tensor of its first one's spec, that spec and the list's length are what hashing each
// tensor would determine (a tensor hashes as its storage kind and spec), found with a comparison per expert rather
// than a hash. Any other list is hashed whole.
ttsl::hash::hash_t hash_expert_list(const std::vector<Tensor>& list) {
    const bool uniform = !list.empty() && std::all_of(list.begin(), list.end(), [&](const Tensor& tensor) {
        return tensor.storage_type() == StorageType::DEVICE && tensor.tensor_spec() == list.front().tensor_spec();
    });
    if (uniform) {
        return ttsl::hash::hash_objects_with_default_seed(true, list.size(), list.front());
    }
    return ttsl::hash::hash_objects_with_default_seed(false, list);
}

}  // namespace

// The key the framework would derive, with no collision-check companion: that one is a string re-encoding every
// attribute and the specs of some thirty tensors, rebuilt on every call, and a 64-bit hash that collides with this
// op's few cache entries is not the risk it is worth paying for. Every input is hashed, the per-expert lists as
// hash_expert_list keys them.
ttsl::hash::hash_t HybridRoutedExpertFfnDeviceOperation::compute_program_hash(
    const operation_attributes_t& op, const tensor_args_t& t) {
    // Binding every field is what keeps this list complete: an input added to HybridRoutedExpertFfnInputs no
    // longer compiles here until it is hashed too.
    const auto& [x, gate_projs, up_projs, down_projs, counts, global_expert_idx_table, output, expert_region_offsets, gate_biases, up_biases, down_biases, l1_arena, dispatched_metadata, expert_offsets, replicated_global_expert_idx_table, forwarding_buffer] =
        t;
    return ttsl::hash::hash_objects_with_default_seed(
        op,
        x,
        hash_expert_list(gate_projs),
        hash_expert_list(up_projs),
        hash_expert_list(down_projs),
        counts,
        global_expert_idx_table,
        output,
        expert_region_offsets,
        hash_expert_list(gate_biases),
        hash_expert_list(up_biases),
        hash_expert_list(down_biases),
        l1_arena,
        dispatched_metadata,
        expert_offsets,
        replicated_global_expert_idx_table,
        forwarding_buffer);
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
