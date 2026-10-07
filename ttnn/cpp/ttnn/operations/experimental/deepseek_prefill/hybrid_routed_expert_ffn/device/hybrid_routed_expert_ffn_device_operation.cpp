// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn_device_operation.hpp"

#include "hybrid_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

void HybridRoutedExpertFfnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& op, const tensor_args_t& t) {
    validate_arguments(op, t);
}

void HybridRoutedExpertFfnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& op, const tensor_args_t& t) {
    // Tensor specs and compile-time attributes participate in the cache key, but buffer addresses
    // do not. Re-run validation so an address-only hit cannot alias the output onto x, move an
    // argument to another device, or hand a half a buffer the cached program was not built for.
    validate_arguments(op, t);
}

HybridRoutedExpertFfnDeviceOperation::spec_return_value_t HybridRoutedExpertFfnDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.output.tensor_spec();
}

HybridRoutedExpertFfnDeviceOperation::tensor_return_value_t HybridRoutedExpertFfnDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.output;
}

tt::tt_metal::ProgramDescriptor HybridRoutedExpertFfnDeviceOperation::create_descriptor(
    const operation_attributes_t& op, const tensor_args_t& t, tensor_return_value_t& output) {
    return create_hybrid_program_descriptor(op, t, output);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
