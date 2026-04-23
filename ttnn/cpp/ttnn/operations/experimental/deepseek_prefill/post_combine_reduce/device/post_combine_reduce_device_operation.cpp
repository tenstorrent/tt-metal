// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "post_combine_reduce_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

void PostCombineReduceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    const ttnn::Tensor& weights = tensor_args.weights;
    const ttnn::Tensor& indices = tensor_args.indices;
    const ttnn::Tensor& expert_dispatch_table = tensor_args.expert_dispatch_table;

    TT_FATAL(combine_output.storage_type() == StorageType::DEVICE, "combine_output must be on device");
    TT_FATAL(combine_output.buffer() != nullptr, "combine_output must have a buffer");
    TT_FATAL(weights.storage_type() == StorageType::DEVICE, "weights must be on device");
    TT_FATAL(weights.buffer() != nullptr, "weights must have a buffer");
    TT_FATAL(indices.storage_type() == StorageType::DEVICE, "indices must be on device");
    TT_FATAL(indices.buffer() != nullptr, "indices must have a buffer");
    TT_FATAL(expert_dispatch_table.storage_type() == StorageType::DEVICE, "expert_dispatch_table must be on device");
    TT_FATAL(expert_dispatch_table.buffer() != nullptr, "expert_dispatch_table must have a buffer");

    const auto combine_rank = combine_output.padded_shape().rank();
    TT_FATAL(
        combine_rank >= 2,
        "combine_output rank must be at least 2 (expert + embedding dimensions), got {}",
        combine_rank);
    TT_FATAL(
        operation_attributes.expert_dim < combine_rank,
        "expert_dim {} must be less than combine_output rank {}",
        operation_attributes.expert_dim,
        combine_rank);

    const auto& combine_shape = combine_output.padded_shape();
    const auto& weights_shape = weights.padded_shape();
    const auto weights_rank = weights_shape.rank();
    TT_FATAL(
        weights_rank == combine_rank,
        "weights rank must match combine_output rank: got weights rank {} and combine_output rank {}",
        weights_rank,
        combine_rank);
    // All dimensions before the embedding dim must match
    for (uint32_t dim = 0; dim < combine_rank - 1; ++dim) {
        TT_FATAL(
            weights_shape[dim] == combine_shape[dim],
            "weights padded_shape[{}] ({}) must match combine_output padded_shape[{}] ({})",
            dim,
            weights_shape[dim],
            dim,
            combine_shape[dim]);
    }
}

void PostCombineReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    const ttnn::Tensor& weights = tensor_args.weights;
    const ttnn::Tensor& indices = tensor_args.indices;
    const ttnn::Tensor& expert_dispatch_table = tensor_args.expert_dispatch_table;

    TT_FATAL(combine_output.layout() == ttnn::Layout::ROW_MAJOR, "combine_output must be ROW_MAJOR");
    TT_FATAL(combine_output.dtype() == DataType::BFLOAT16, "combine_output must be bfloat16");
    TT_FATAL(weights.layout() == ttnn::Layout::ROW_MAJOR, "weights must be ROW_MAJOR");
    TT_FATAL(weights.dtype() == DataType::BFLOAT16, "weights must be bfloat16");
    TT_FATAL(indices.layout() == ttnn::Layout::ROW_MAJOR, "indices must be ROW_MAJOR");
    TT_FATAL(indices.dtype() == DataType::INT32, "indices must be int32");
    TT_FATAL(expert_dispatch_table.layout() == ttnn::Layout::ROW_MAJOR, "expert_dispatch_table must be ROW_MAJOR");
    TT_FATAL(expert_dispatch_table.dtype() == DataType::INT32, "expert_dispatch_table must be int32");
}

ttnn::TensorSpec PostCombineReduceDeviceOperation::compute_output_specs(
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

    // Use TILE layout for hardware tilization output
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            combine_output.dtype(), tt::tt_metal::PageConfig(Layout::TILE), output_memory_config));
}

ttnn::Tensor PostCombineReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.combine_output.device());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce

namespace ttnn::prim {

ttnn::Tensor post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& expert_dispatch_table,
    uint32_t expert_dim,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    namespace pcr = ttnn::operations::experimental::deepseek_prefill::post_combine_reduce;
    using OperationType = pcr::PostCombineReduceDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        pcr::PostCombineReduceParams{expert_dim, output_memory_config},
        pcr::PostCombineReduceInputs{combine_output, weights, indices, expert_dispatch_table});
}

}  // namespace ttnn::prim
