// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads {

SplitFusedQKVAndSplitHeadsDeviceOperation::program_factory_t
SplitFusedQKVAndSplitHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input_tensor.is_sharded()) {
        return program::SplitFusedQKVAndSplitHeadsShardedProgramFactory{};
    }
    return program::SplitFusedQKVAndSplitHeadsProgramFactory{};
}

void SplitFusedQKVAndSplitHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void SplitFusedQKVAndSplitHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensors = tensor_args.output_tensors;
    const auto batch_size = input_tensor.padded_shape()[0];

    // TODO: See issue #1744
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");

    if (!input_tensor.is_sharded()) {
        TT_FATAL(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 7 to 9 for bert large TM ops!");
    } else {
        auto bbox = input_tensor.shard_spec().value().grid.bounding_box();
        TT_FATAL(
            (bbox.end_coord.x < operation_attributes.compute_with_storage_grid_size.x &&
             bbox.end_coord.y < operation_attributes.compute_with_storage_grid_size.y),
            "Bounding box end coordinates ({}, {}) must be less than grid size ({}, {})",
            bbox.end_coord.x,
            bbox.end_coord.y,
            operation_attributes.compute_with_storage_grid_size.x,
            operation_attributes.compute_with_storage_grid_size.y);
        TT_FATAL(
            input_tensor.shard_spec().value().grid.ranges().size() == 1,
            "Input tensor shard spec must have exactly 1 grid range but got {}",
            input_tensor.shard_spec().value().grid.ranges().size());
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            "Input tensor memory layout must be BLOCK_SHARDED but got {}",
            input_tensor.memory_config().memory_layout());
    }

    if (!output_tensors.empty()) {
        TT_FATAL(output_tensors.size() == 3, "Must have 3 output tensors");
    }
}

SplitFusedQKVAndSplitHeadsDeviceOperation::spec_return_value_t
SplitFusedQKVAndSplitHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::Layout;
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensors = tensor_args.output_tensors;

    if (output_tensors.size() == 3 && output_tensors[0].has_value() && output_tensors[1].has_value() &&
        output_tensors[2].has_value()) {
        return {
            output_tensors.at(0)->tensor_spec(),
            output_tensors.at(1)->tensor_spec(),
            output_tensors.at(2)->tensor_spec()};
    }

    const auto batch_size = input_tensor.padded_shape()[0];
    uint32_t num_heads = operation_attributes.num_heads;
    uint32_t num_output_tensors = 3;
    uint32_t M = input_tensor.padded_shape()[2];                                    // 384
    uint32_t K = input_tensor.padded_shape()[-1] / num_output_tensors / num_heads;  // 64

    if (input_tensor.is_sharded()) {
        // core range
        CoreRangeSet all_cores = input_tensor.shard_spec().value().grid;
        tt::tt_metal::ShardOrientation shard_orientation = input_tensor.shard_spec().value().orientation;
        auto bbox = all_cores.bounding_box();
        uint32_t num_M_cores = shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR ? bbox.end_coord.x + 1
                                                                                              : bbox.end_coord.y + 1;
        // shard spec
        uint32_t per_core_M_qv = (num_heads / num_M_cores) * M;  // 768
        uint32_t per_core_N_qv = K;                              // 64
        auto shard_spec_qv = tt::tt_metal::ShardSpec{all_cores, {per_core_M_qv, per_core_N_qv}, shard_orientation};
        uint32_t per_core_M_k = (num_heads / num_M_cores) * K;  // 128
        uint32_t per_core_N_k = M;                              // 384
        auto shard_spec_k = tt::tt_metal::ShardSpec{all_cores, {per_core_M_k, per_core_N_k}, shard_orientation};
        // create sharded tensors
        auto mem_config_qv = operation_attributes.output_mem_config.with_shard_spec(shard_spec_qv);
        auto mem_config_k = operation_attributes.output_mem_config.with_shard_spec(shard_spec_k);
        auto out_tensor_q = TensorSpec(
            Shape({batch_size, num_heads, M, K}),
            TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), mem_config_qv));
        auto out_tensor_k = TensorSpec(
            Shape({batch_size, num_heads, K, M}),
            TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), mem_config_k));
        auto out_tensor_v = TensorSpec(
            Shape({batch_size, num_heads, M, K}),
            TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), mem_config_qv));
        return {out_tensor_q, out_tensor_k, out_tensor_v};
    }

    TensorLayout layout(input_tensor.dtype(), PageConfig(Layout::TILE), operation_attributes.output_mem_config);
    return {
        TensorSpec(Shape({batch_size, num_heads, M, K}), layout),
        TensorSpec(Shape({batch_size, num_heads, K, M}), layout),
        TensorSpec(Shape({batch_size, num_heads, M, K}), layout),
    };
}

SplitFusedQKVAndSplitHeadsDeviceOperation::tensor_return_value_t
SplitFusedQKVAndSplitHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(specs[0], tensor_args.input_tensor.device()),
        create_device_tensor(specs[1], tensor_args.input_tensor.device()),
        create_device_tensor(specs[2], tensor_args.input_tensor.device()),
    };
}

}  // namespace ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads

namespace ttnn::prim {

std::vector<Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    uint32_t num_heads,
    const std::optional<std::vector<std::optional<ttnn::Tensor>>>& optional_output_tensors) {
    using OperationType = ttnn::operations::experimental::transformer::split_query_key_value_and_split_heads::
        SplitFusedQKVAndSplitHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        compute_with_storage_grid_size, memory_config.value_or(input_tensor.memory_config()), num_heads};
    auto tensor_args = OperationType::tensor_args_t{
        input_tensor, optional_output_tensors.value_or(std::vector<std::optional<ttnn::Tensor>>{})};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
