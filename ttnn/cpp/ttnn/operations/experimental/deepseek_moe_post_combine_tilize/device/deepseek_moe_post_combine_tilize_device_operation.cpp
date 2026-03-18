// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {
void DeepseekMoEPostCombineTilizeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}

void DeepseekMoEPostCombineTilizeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    uint32_t upper_dims = 1;
    for (uint32_t dim = 0; dim < input_rank - 1; ++dim) {
        upper_dims *= input_shape[dim];
    }

    const tt::tt_metal::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;

    // rank 2+
    TT_FATAL(input_rank >= 2, "DeepseekMoEPostCombineTilize requires rank >= 2, but has {}", input_rank);

    // input must be bfloat16
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16,
        "DeepseekMoEPostCombineTilize requires dtype to be bfloat16, but has {}",
        input_tensor.dtype());

    // input must be interleaved
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "DeepseekMoEPostCombineTilize requires input to be interleaved");

    // input must be row-major layout (kernel reads rows as pages)
    TT_FATAL(
        input_tensor.layout() == ttnn::Layout::ROW_MAJOR,
        "DeepseekMoEPostCombineTilize requires input layout to be ROW_MAJOR");

    // output must be L1 sharded
    TT_FATAL(
        output_memory_config.is_sharded() && output_memory_config.nd_shard_spec().has_value(),
        "DeepseekMoEPostCombineTilize requires sharded output");
    TT_FATAL(output_memory_config.buffer_type() == BufferType::L1, "DeepseekMoEPostCombineTilize requires L1 output");

    auto output_nd_shard_spec = output_memory_config.nd_shard_spec().value();
    auto output_nd_shard_shape = output_nd_shard_spec.shard_shape;
    uint32_t output_shard_width = output_nd_shard_shape[-1];
    uint32_t output_shard_height = output_nd_shard_shape[-2];

    // must be a 2d shard shape
    TT_FATAL(output_nd_shard_shape.rank() == 2, "DeepseekMoEPostCombineTilize requires dimension 2 output shard shape");

    // output shard width must be even multiple of tiles and even split tensor width
    TT_FATAL(
        output_shard_width % tt::constants::TILE_WIDTH == 0 && input_shape[-1] % output_shard_width == 0,
        "DeepseekMoEPostCombineTilize requires output shard width to be even multiple of tiles");

    // output shard height must be single tile high and even split physical tensor height
    TT_FATAL(
        output_shard_height == tt::constants::TILE_HEIGHT && upper_dims % output_shard_height == 0,
        "DeepseekMoEPostCombineTilize requires output shard height to be a single tile high");

    // single shard per core
    uint32_t num_shards_wide = input_shape[-1] / output_shard_width;
    uint32_t num_shards_high = upper_dims / output_shard_height;
    TT_FATAL(
        num_shards_wide * num_shards_high == output_nd_shard_spec.grid.num_cores(),
        "DeepseekMoEPostCombineTilize requires one output shard per core");
}

ttnn::TensorSpec DeepseekMoEPostCombineTilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const auto& output_shape = input_tensor.padded_shape();

    const tt::tt_metal::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;

    return TensorSpec(
        output_shape,
        operations::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), output_memory_config));
}

ttnn::Tensor DeepseekMoEPostCombineTilizeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const ttnn::TensorSpec& output_tensor_spec = compute_output_specs(operation_attributes, tensor_args);

    return create_device_tensor(output_tensor_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor deepseek_moe_post_combine_tilize(
    const ttnn::Tensor& input_tensor, const tt::tt_metal::MemoryConfig& output_memory_config) {
    using OperationType = ttnn::experimental::prim::DeepseekMoEPostCombineTilizeDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{output_memory_config}, OperationType::tensor_args_t{input_tensor});
}

}  // namespace ttnn::prim
