// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::mla {

MatmulWODeviceOperation::program_factory_t MatmulWODeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::MatmulWOProgramFactory{};
}

void MatmulWODeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MatmulWODeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.input_tensor.logical_shape().rank() >= 2,
        "Input tensor must be at least rank 2, got {}",
        tensor_args.input_tensor.logical_shape().rank());
}

MatmulWODeviceOperation::spec_return_value_t MatmulWODeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Use the output tensor's spec since it's passed in with the correct sharded memory config
    const auto& output_tensor = tensor_args.output_tensor;
    return output_tensor.tensor_spec();
}

MatmulWODeviceOperation::tensor_return_value_t MatmulWODeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Return the preallocated output tensor (already sharded with correct memory config)
    return tensor_args.output_tensor;
}

std::tuple<MatmulWODeviceOperation::operation_attributes_t, MatmulWODeviceOperation::tensor_args_t>
MatmulWODeviceOperation::invoke(
    const Tensor& input_tensor, const Tensor& w_tensor, const Tensor& output_tensor, const uint32_t layer_id) {
    return {
        operation_attributes_t{.layer_id = layer_id},
        tensor_args_t{.input_tensor = input_tensor, .w_tensor = w_tensor, .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::deepseek::mla
