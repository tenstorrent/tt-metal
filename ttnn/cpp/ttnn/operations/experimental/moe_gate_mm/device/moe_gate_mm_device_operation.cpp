// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gate_mm_device_operation.hpp"

namespace ttnn::operations::experimental::moe_gate_mm {

MoEGateMMDeviceOperation::program_factory_t MoEGateMMDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MoEGateMMProgramFactory{};
}

void MoEGateMMDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEGateMMDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.logical_shape().rank() >= 2,
        "Input tensor must be at least rank 2, got {}",
        tensor_args.input_tensor.logical_shape().rank());
    TT_FATAL(args.layer_id >= 0, "Layer ID must be non-negative, got {}", args.layer_id);
}

MoEGateMMDeviceOperation::spec_return_value_t MoEGateMMDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Use the output tensor's spec since it's passed in with the correct sharded memory config
    const auto& output_tensor = tensor_args.output_tensor;
    return output_tensor.tensor_spec();
}

MoEGateMMDeviceOperation::tensor_return_value_t MoEGateMMDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Return the preallocated output tensor (already sharded with correct memory config)
    return tensor_args.output_tensor;
}

std::tuple<MoEGateMMDeviceOperation::operation_attributes_t, MoEGateMMDeviceOperation::tensor_args_t>
MoEGateMMDeviceOperation::invoke(
    const Tensor& input_tensor, const Tensor& w_tensor, const Tensor& output_tensor, const uint32_t layer_id) {
    return {
        operation_attributes_t{.layer_id = layer_id},
        tensor_args_t{.input_tensor = input_tensor, .w_tensor = w_tensor, .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::moe_gate_mm
