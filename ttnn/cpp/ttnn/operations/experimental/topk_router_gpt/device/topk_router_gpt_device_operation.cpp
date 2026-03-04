// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_device_operation.hpp"

namespace ttnn::operations::experimental::topk_router_gpt {

void TopkRouterGptDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

void TopkRouterGptDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {
    // Validation is intentionally minimal for performance.
    // The Python wrapper ensures correct tensor shapes and dtypes.
}

spec_return_value_t TopkRouterGptDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

tensor_return_value_t TopkRouterGptDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor;
}

std::tuple<TopkRouterGptDeviceOperation::operation_attributes_t, TopkRouterGptDeviceOperation::tensor_args_t>
TopkRouterGptDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const Tensor& bias_tensor,
    const Tensor& output_tensor,
    uint32_t k,
    uint32_t num_experts,
    bool untilize_output) {
    return {
        operation_attributes_t{k, num_experts, untilize_output},
        tensor_args_t{input_tensor, weight_tensor, bias_tensor, output_tensor}};
}

}  // namespace ttnn::operations::experimental::topk_router_gpt
