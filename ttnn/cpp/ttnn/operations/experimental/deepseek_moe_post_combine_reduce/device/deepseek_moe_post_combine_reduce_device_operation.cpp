// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "deepseek_moe_post_combine_reduce_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

void DeepseekMoEPostCombineReduceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    TT_FATAL(combine_output.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(combine_output.buffer() != nullptr, "Input must have a buffer");
}

void DeepseekMoEPostCombineReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    const ttnn::Tensor& weights = tensor_args.weights;

    // Basic validations
    TT_FATAL(combine_output.layout() == ttnn::Layout::ROW_MAJOR, "combine_output must be ROW_MAJOR");
    TT_FATAL(combine_output.dtype() == DataType::BFLOAT16, "combine_output must be bfloat16");
    TT_FATAL(weights.storage_type() == StorageType::DEVICE, "weights must be on device");
}

ttnn::TensorSpec DeepseekMoEPostCombineReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    const auto& input_shape = combine_output.padded_shape();

    // Output shape: remove expert dimension
    // Input: [B, C, seq_len, num_experts, emb_dim]
    // Output: [B, C, seq_len, emb_dim]
    std::vector<uint32_t> output_shape_vec;
    for (uint32_t i = 0; i < input_shape.rank(); ++i) {
        if (i != operation_attributes.expert_dim) {
            output_shape_vec.push_back(input_shape[i]);
        }
    }

    const ttnn::Shape output_shape(output_shape_vec);
    const tt::tt_metal::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;

    // Use ROW_MAJOR layout to match our fake tilization approach
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            combine_output.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), output_memory_config));
}

ttnn::Tensor DeepseekMoEPostCombineReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.combine_output.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor deepseek_moe_post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    uint32_t expert_dim,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using OperationType = ttnn::experimental::prim::DeepseekMoEPostCombineReduceDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        ttnn::experimental::prim::DeepseekMoEPostCombineReduceParams{expert_dim, output_memory_config},
        ttnn::experimental::prim::DeepseekMoEPostCombineReduceInputs{combine_output, weights});
}

}  // namespace ttnn::prim
