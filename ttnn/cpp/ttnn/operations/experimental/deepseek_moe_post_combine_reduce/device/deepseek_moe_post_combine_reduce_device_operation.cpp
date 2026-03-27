// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_reduce/device/deepseek_moe_post_combine_reduce_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_reduce/device/deepseek_moe_post_combine_reduce_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

void DeepseekMoEPostCombineReduceDeviceOperationImpl::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void DeepseekMoEPostCombineReduceDeviceOperationImpl::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {

    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    const ttnn::Tensor& weights = tensor_args.weights;

    // Validate combine_output tensor
    TT_FATAL(
        combine_output.storage_type() == StorageType::DEVICE,
        "Combine output tensor must be on device");
    TT_FATAL(
        combine_output.buffer() != nullptr,
        "Combine output tensor must have a buffer");
    TT_FATAL(
        combine_output.dtype() == DataType::BFLOAT16,
        "Combine output tensor must be bfloat16");
    TT_FATAL(
        combine_output.layout() == ttnn::Layout::ROW_MAJOR,
        "Combine output tensor must be ROW_MAJOR layout");

    // Validate weights tensor
    TT_FATAL(
        weights.storage_type() == StorageType::DEVICE,
        "Weights tensor must be on device");
    TT_FATAL(
        weights.buffer() != nullptr,
        "Weights tensor must have a buffer");
    TT_FATAL(
        weights.dtype() == DataType::BFLOAT16,
        "Weights tensor must be bfloat16");

    // Validate tensor shapes
    const auto& combine_shape = combine_output.padded_shape();
    const auto& weights_shape = weights.padded_shape();

    TT_FATAL(
        combine_shape.rank() >= 3,
        "Combine output must have at least 3 dimensions");
    TT_FATAL(
        operation_attributes.expert_dim < combine_shape.rank(),
        "Expert dimension {} must be less than tensor rank {}",
        operation_attributes.expert_dim, combine_shape.rank());

    // Validate that weight shape matches combine output (excluding expert and embedding dims)
    // combine_output: [..., seq_len, num_experts, emb_dim]
    // weights: [..., seq_len, num_experts]
    for (uint32_t i = 0; i < weights_shape.rank(); ++i) {
        TT_FATAL(
            weights_shape[i] == combine_shape[i],
            "Weight shape mismatch at dimension {}: weights={}, combine_output={}",
            i, weights_shape[i], combine_shape[i]);
    }
}

ttnn::TensorSpec DeepseekMoEPostCombineReduceDeviceOperationImpl::compute_output_specs(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {

    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    const auto& input_shape = combine_output.padded_shape();

    // Output shape: remove the expert dimension
    // Input:  [..., seq_len, num_experts, emb_dim]
    // Output: [..., seq_len, emb_dim]
    ttnn::SmallVector<uint32_t> output_shape_vec;
    for (uint32_t i = 0; i < input_shape.rank(); ++i) {
        if (i != operation_attributes.expert_dim) {
            output_shape_vec.push_back(input_shape[i]);
        }
    }

    const ttnn::SimpleShape output_shape(output_shape_vec);

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            combine_output.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),  // Output in TILE_LAYOUT
            operation_attributes.output_memory_config
        )
    );
}

ttnn::Tensor DeepseekMoEPostCombineReduceDeviceOperationImpl::create_output_tensors(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args) {

    const ttnn::TensorSpec& output_tensor_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_tensor_spec, tensor_args.combine_output.device());
}

DeepseekMoEPostCombineReduceDeviceOperationImpl::program_factory_t
DeepseekMoEPostCombineReduceDeviceOperationImpl::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return DeepseekMoEPostCombineReduceProgramFactory{};
}

tt::stl::reflection::Attributes DeepseekMoEPostCombineReduceDeviceOperationImpl::attributes() {
    return {};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor deepseek_moe_post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    const uint32_t expert_dim,
    const tt::tt_metal::MemoryConfig& output_memory_config) {

    using OperationType = ttnn::experimental::prim::DeepseekMoEPostCombineReduceDeviceOperationImpl;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .expert_dim = expert_dim,
            .output_memory_config = output_memory_config
        },
        OperationType::tensor_args_t{
            .combine_output = combine_output,
            .weights = weights
        }
    );
}

}  // namespace ttnn::prim