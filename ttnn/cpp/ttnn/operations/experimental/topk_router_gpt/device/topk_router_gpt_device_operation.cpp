// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_router_gpt_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

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
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto input_shape = tensor_args.input_tensor.logical_shape();
    auto B = input_shape[0];
    auto dram_rm = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    // Slot 0: indices_rm [B, k_padded] uint16 RM
    uint32_t k_padded = ((attrs.k + 7) / 8) * 8;
    auto idx_spec =
        TensorSpec(ttnn::Shape({B, k_padded}), TensorLayout(DataType::UINT16, PageConfig(Layout::ROW_MAJOR), dram_rm));

    // Slot 1: weights_rm [B, k_padded] bf16 RM
    auto wgt_spec = TensorSpec(
        ttnn::Shape({B, k_padded}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), dram_rm));

    return {idx_spec, wgt_spec};
}

tensor_return_value_t TopkRouterGptDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attrs, tensor_args);
    auto device = tensor_args.input_tensor.device();
    auto idx_tensor = create_device_tensor(std::get<0>(specs), device);
    auto wgt_tensor = create_device_tensor(std::get<1>(specs), device);
    return {idx_tensor, wgt_tensor};
}

std::tuple<TopkRouterGptDeviceOperation::operation_attributes_t, TopkRouterGptDeviceOperation::tensor_args_t>
TopkRouterGptDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const Tensor& bias_tensor,
    uint32_t k,
    uint32_t num_experts) {
    return {operation_attributes_t{k, num_experts}, tensor_args_t{input_tensor, weight_tensor, bias_tensor}};
}

}  // namespace ttnn::operations::experimental::topk_router_gpt
