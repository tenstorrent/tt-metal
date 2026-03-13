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

    const tt::tt_metal::MemoryConfig& output_memory_config = operation_attributes.output_memory_config;

    // rank 2+
    TT_FATAL(input_rank >= 2, "DeepseekMoEPostCombineTilize requires rank >= 2, but has {}", input_rank);

    // enough cores for even split among tile height
    uint32_t upper_dims = 1;
    for (uint32_t dim = 0; dim < input_rank - 1; ++dim) {
        upper_dims *= input_shape[dim];
    }
    uint32_t total_tile_height = upper_dims / tt::constants::TILE_HEIGHT;

    auto grid_size = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;
    TT_FATAL(
        total_tile_height <= total_cores,
        "DeepseekMoEPostCombineTilize requires total tile height to be less than or equal to the total number of "
        "cores, but has {}",
        total_tile_height);

    // input must be interleaved
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16,
        "DeepseekMoEPostCombineTilize requires dtype to be bfloat16, but has {}",
        input_tensor.dtype());

    // input must be bfloat16
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "DeepseekMoEPostCombineTilize requires input to be interleaved");

    // if output is sharded, shard shape must be even multiple of tiles
    if (output_memory_config.is_sharded()) {
        auto shard_spec = input_tensor.memory_config().shard_spec().value();
        uint32_t shard_width = shard_spec.shape[-1];
        uint32_t shard_height = shard_spec.shape[-2];

        TT_FATAL(
            shard_width % tt::constants::TILE_WIDTH == 0,
            "DeepseekMoEPostCombineTilize requires output shard shape to be even multiple of tiles");
        TT_FATAL(
            shard_height % tt::constants::TILE_HEIGHT == 0,
            "DeepseekMoEPostCombineTilize requires output shard shape to be even multiple of tiles");
    }
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
    const ttnn::Tensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using OperationType = ttnn::experimental::prim::DeepseekMoEPostCombineTilizeDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{output_memory_config.value_or(input_tensor.memory_config())},
        OperationType::tensor_args_t{input_tensor});
}

}  // namespace ttnn::prim
