// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_device_operation.hpp"

namespace ttnn::operations::experimental::moe {

MoEDeviceOperation::program_factory_t MoEDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoEProgramFactory{};
}

void MoEDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.logical_shape().rank() >= 2,
        "Input tensor must be at least rank 2, got {}",
        tensor_args.input_tensor.logical_shape().rank());
    TT_FATAL(args.num_experts >= 1, "Number of experts must be at least 1, got {}", args.num_experts);
}

MoEDeviceOperation::spec_return_value_t MoEDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Use the output tensor's spec since it's passed in with the correct sharded memory config
    const auto& output_tensor = tensor_args.output_tensor;
    return output_tensor.tensor_spec();
}

MoEDeviceOperation::tensor_return_value_t MoEDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Return the preallocated output tensor (already sharded with correct memory config)
    return tensor_args.output_tensor;
}

std::tuple<MoEDeviceOperation::operation_attributes_t, MoEDeviceOperation::tensor_args_t> MoEDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& w0_w1_tensor,
    const Tensor& w2_tensor,
    const Tensor& output_tensor,
    const uint32_t num_experts,
    const uint32_t layer_id,
    const tt::tt_metal::CoreRangeSet& output_shard_core_ranges) {
    return {
        operation_attributes_t{
            .num_experts = num_experts, .layer_id = layer_id, .output_shard_core_ranges = output_shard_core_ranges},
        tensor_args_t{
            .input_tensor = input_tensor,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor,
            .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::moe
