// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_wqkv_ab_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab {

void MlaWqkvAbDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MlaWqkvAbDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

MlaWqkvAbDeviceOperation::spec_return_value_t MlaWqkvAbDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Use the output tensor's spec since it's passed in with the correct sharded memory config
    const auto& output_tensor = tensor_args.output_tensor;
    return output_tensor.tensor_spec();
}

MlaWqkvAbDeviceOperation::tensor_return_value_t MlaWqkvAbDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Return the preallocated output tensor (already sharded with correct memory config)
    return tensor_args.output_tensor;
}

std::tuple<MlaWqkvAbDeviceOperation::operation_attributes_t, MlaWqkvAbDeviceOperation::tensor_args_t>
MlaWqkvAbDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& w_a_tensor,
    const Tensor& wq_b_tensor,
    const Tensor& q_nope_tensor,
    const Tensor& rope_tensor,
    const Tensor& output_tensor,
    const uint32_t layer_id,
    const uint32_t pos) {
    return {
        operation_attributes_t{.layer_id = layer_id, .pos = pos},
        tensor_args_t{
            .input_tensor = input_tensor,
            .w_a_tensor = w_a_tensor,
            .wq_b_tensor = wq_b_tensor,
            .q_nope_tensor = q_nope_tensor,
            .rope_tensor = rope_tensor,
            .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab
