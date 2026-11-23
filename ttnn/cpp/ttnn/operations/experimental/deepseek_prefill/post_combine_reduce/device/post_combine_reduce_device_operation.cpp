// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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
    const auto& indices = tensor_args.indices;
    const auto& expert_dispatch_table = tensor_args.expert_dispatch_table;

    TT_FATAL(combine_output.storage_type() == StorageType::DEVICE, "combine_output must be on device");
    TT_FATAL(combine_output.is_allocated(), "combine_output must have a buffer");
    TT_FATAL(weights.storage_type() == StorageType::DEVICE, "weights must be on device");
    TT_FATAL(weights.is_allocated(), "weights must have a buffer");

    // indices and expert_dispatch_table must be supplied together or not at all:
    // both provided  -> DeepSeek dispatch-table-based expert skip
    // both omitted   -> GPT-OSS weight==0 skip
    TT_FATAL(
        indices.has_value() == expert_dispatch_table.has_value(),
        "indices and expert_dispatch_table must be provided together or omitted together; "
        "got indices={}, expert_dispatch_table={}",
        indices.has_value(),
        expert_dispatch_table.has_value());

    if (indices.has_value()) {
        TT_FATAL(indices->storage_type() == StorageType::DEVICE, "indices must be on device");
        TT_FATAL(indices->is_allocated(), "indices must have a buffer");
        TT_FATAL(
            expert_dispatch_table->storage_type() == StorageType::DEVICE, "expert_dispatch_table must be on device");
        TT_FATAL(expert_dispatch_table->is_allocated(), "expert_dispatch_table must have a buffer");
    }

    if (tensor_args.combine_row_index.has_value()) {
        // Checked on every call: the reader takes the index buffer's address as a runtime arg.
        TT_FATAL(
            tensor_args.combine_row_index->device() == combine_output.device(),
            "combine_row_index must be on combine_output's device");
    }

    const ttnn::Shape combine_shape = logical_combine_shape(tensor_args);
    const auto combine_rank = combine_shape.rank();
    TT_FATAL(
        combine_rank >= 2,
        "combine_output rank must be at least 2 (expert + embedding dimensions), got {}",
        combine_rank);
    TT_FATAL(
        operation_attributes.expert_dim < combine_rank,
        "expert_dim {} must be less than combine_output rank {}",
        operation_attributes.expert_dim,
        combine_rank);

    const auto& weights_shape = weights.padded_shape();
    const auto weights_rank = weights_shape.rank();
    // Two weight layouts: one weight per page ([..., num_experts, 1], rank == combine rank), or one token's weights
    // per page ([..., num_experts], rank == combine rank - 1, expert dim last) -- read once per token.
    TT_FATAL(
        weights_rank == combine_rank || weights_rank + 1 == combine_rank,
        "weights rank must match combine_output rank (or be one less): got weights rank {} and combine_output rank {}",
        weights_rank,
        combine_rank);
    TT_FATAL(
        weights_rank == combine_rank || operation_attributes.expert_dim + 2 == combine_rank,
        "per-token weights ([..., num_experts]) need the expert dim next to the embedding dim, got expert_dim {}",
        operation_attributes.expert_dim);
    // All dimensions before the embedding dim must match
    for (uint32_t dim = 0; dim < std::min<uint32_t>(weights_rank, combine_rank - 1); ++dim) {
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
    const auto& indices = tensor_args.indices;
    const auto& expert_dispatch_table = tensor_args.expert_dispatch_table;

    TT_FATAL(combine_output.layout() == ttnn::Layout::ROW_MAJOR, "combine_output must be ROW_MAJOR");
    TT_FATAL(combine_output.dtype() == DataType::BFLOAT16, "combine_output must be bfloat16");
    TT_FATAL(weights.layout() == ttnn::Layout::ROW_MAJOR, "weights must be ROW_MAJOR");
    TT_FATAL(weights.dtype() == DataType::BFLOAT16, "weights must be bfloat16");
    if (weights.padded_shape().rank() + 1 == logical_combine_shape(tensor_args).rank()) {
        // The writer reads one page per token and takes all of the token's weights from it.
        TT_FATAL(
            weights.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "per-token weights must be interleaved (one page holds one token's weights)");
    }

    if (indices.has_value()) {
        TT_FATAL(indices->layout() == ttnn::Layout::ROW_MAJOR, "indices must be ROW_MAJOR");
        // #44928: indices come straight from moe_grouped_topk as UINT16. The
        // writer expands them to int32 in L1 before the compute kernel reads.
        TT_FATAL(
            indices->dtype() == DataType::UINT16,
            "indices must be uint16 (matching moe_grouped_topk output), got {}",
            indices->dtype());
        TT_FATAL(expert_dispatch_table->layout() == ttnn::Layout::ROW_MAJOR, "expert_dispatch_table must be ROW_MAJOR");
        TT_FATAL(expert_dispatch_table->dtype() == DataType::INT32, "expert_dispatch_table must be int32");
    }

    if (tensor_args.combine_row_index.has_value()) {
        const auto& ri = *tensor_args.combine_row_index;
        TT_FATAL(
            ri.storage_type() == StorageType::DEVICE && ri.buffer() != nullptr, "combine_row_index must be on device");
        TT_FATAL(ri.dtype() == DataType::UINT32, "combine_row_index must be uint32, got {}", ri.dtype());
        TT_FATAL(ri.layout() == ttnn::Layout::ROW_MAJOR, "combine_row_index must be ROW_MAJOR");
        TT_FATAL(
            ri.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "combine_row_index must be interleaved");
        // The reader reads row combine_row_index[t, j] as one whole page of the table.
        TT_FATAL(
            combine_output.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "with combine_row_index, combine_output must be interleaved (one page per row)");
        TT_FATAL(
            combine_output.padded_shape().rank() == 2,
            "with combine_row_index, combine_output must be a [rows, emb_dim] table, got shape {}",
            combine_output.padded_shape());
        TT_FATAL(
            operation_attributes.expert_dim + 2 == logical_combine_shape(tensor_args).rank(),
            "with combine_row_index, expert_dim ({}) must be its last dim",
            operation_attributes.expert_dim);
    } else {
        TT_FATAL(
            operation_attributes.skip_row == std::numeric_limits<uint32_t>::max(), "skip_row needs combine_row_index");
    }
}

tt::tt_metal::TensorSpec PostCombineReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& combine_output = tensor_args.combine_output;
    const ttnn::Shape input_shape = logical_combine_shape(tensor_args);

    // Output shape: remove expert dimension
    // Input: [B, C, seq_len, num_experts, emb_dim]
    // Output: [B, C, seq_len, emb_dim]
    std::vector<uint32_t> output_shape_vec;
    output_shape_vec.reserve(input_shape.rank());
    for (uint32_t i = 0; i < input_shape.rank(); ++i) {
        if (i != operation_attributes.expert_dim) {
            output_shape_vec.push_back(input_shape[i]);
        }
    }

    const ttnn::Shape output_shape(output_shape_vec);
    const tt::tt_metal::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;

    // Use TILE layout for hardware tilization output
    return tt::tt_metal::TensorSpec(
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
    const std::optional<ttnn::Tensor>& indices,
    const std::optional<ttnn::Tensor>& expert_dispatch_table,
    uint32_t expert_dim,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::Tensor>& combine_row_index,
    uint32_t skip_row) {
    namespace pcr = ttnn::operations::experimental::deepseek_prefill::post_combine_reduce;
    using OperationType = pcr::PostCombineReduceDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        pcr::PostCombineReduceParams{expert_dim, output_memory_config, skip_row},
        pcr::PostCombineReduceInputs{combine_output, weights, indices, expert_dispatch_table, combine_row_index});
}

}  // namespace ttnn::prim
