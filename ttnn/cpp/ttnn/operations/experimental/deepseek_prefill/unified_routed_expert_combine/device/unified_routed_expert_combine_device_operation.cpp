// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_combine_device_operation.hpp"

#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_combine_overlap/device/routed_expert_handoff.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine {

namespace {

namespace overlap = ::ttnn::operations::experimental::deepseek_prefill::routed_expert_combine_overlap;

constexpr const char* kOpName = "unified_routed_expert_moe_combine";

// Below three tiles of tokens the first walk could end in a partial eighth wave, which leaves rows' rings behind.
constexpr uint32_t kMinBigFirstThreshold = 3 * 32;

void validate(const UnifiedRoutedExpertCombineParams& op, const UnifiedRoutedExpertCombineInputs& t) {
    TT_FATAL(
        op.big_first_threshold == 0 || op.big_first_threshold >= kMinBigFirstThreshold,
        "{}: big_first_threshold must be 0 or at least {} tokens, got {}",
        kOpName,
        kMinBigFirstThreshold,
        op.big_first_threshold);
    // The routed expert validates what it will actually be handed, so this op cannot pass a configuration
    // unified_routed_expert_moe would reject on its own.
    ::ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::
        UnifiedRoutedExpertFfnDeviceOperation::validate_on_program_cache_miss(
            routed_expert_attributes(op), routed_expert_inputs(t));
    TT_FATAL(
        t.x.layout() == tt::tt_metal::Layout::ROW_MAJOR && t.x.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "{}: dispatched_buffer must be bfloat16 ROW_MAJOR, got {} {}",
        kOpName,
        t.x.dtype(),
        t.x.layout());
    // Combine's untilizers read the routed expert's output as bfloat8_b tiles, as combine does back to back.
    TT_FATAL(
        t.output.layout() == tt::tt_metal::Layout::TILE && t.output.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "{}: the routed expert writes a bfloat8_b TILE output, got {} {}",
        kOpName,
        t.output.dtype(),
        t.output.layout());
    // The framework resolves cache-hit bindings by buffer, and x and the output are both inputs here.
    TT_FATAL(t.output.buffer() != t.x.buffer(), "{} needs an output distinct from dispatched_buffer", kOpName);
    overlap::validate_overlap_semaphores(
        kOpName,
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
    overlap::validate_forwarding_buffer(kOpName, t.forwarding_buffer, combine_attributes(op, t), combine_inputs(t));
    ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dDeviceOperation::
        validate_on_program_cache_miss(combine_attributes(op, t), combine_inputs(t));
}

}  // namespace

UnifiedRoutedExpertCombineDeviceOperation::program_factory_t
UnifiedRoutedExpertCombineDeviceOperation::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return UnifiedRoutedExpertCombineProgramFactory{};
}

void UnifiedRoutedExpertCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& t) {
    validate(op, t);
}

void UnifiedRoutedExpertCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& op, const tensor_args_t& t) {
    // Tensor specs and compile-time attributes participate in the cache key, but buffer addresses do not. Re-run
    // validation so an address-only hit cannot alias the output onto x or move an argument to another device.
    validate(op, t);
}

UnifiedRoutedExpertCombineDeviceOperation::spec_return_value_t
UnifiedRoutedExpertCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& op, const tensor_args_t& t) {
    return ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dDeviceOperation::
        compute_output_specs(combine_attributes(op, t), combine_inputs(t));
}

UnifiedRoutedExpertCombineDeviceOperation::tensor_return_value_t
UnifiedRoutedExpertCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& op, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(op, t), t.x.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine
