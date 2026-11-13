// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_rmsnorm_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm {

namespace {

using namespace tt::tt_metal;
using namespace tt::constants;

void validate_sharded_rmsnorm_tensors(const ShardedRmsNormParams& attributes, const ShardedRmsNormInputs& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "sharded_rmsnorm: input_tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "sharded_rmsnorm: input_tensor must be allocated on device");
    TT_FATAL(input.device()->arch() == tt::ARCH::BLACKHOLE, "sharded_rmsnorm: Blackhole architecture required");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "sharded_rmsnorm: input_tensor must be BFLOAT16");
    TT_FATAL(
        input.layout() == Layout::TILE || input.layout() == Layout::ROW_MAJOR,
        "sharded_rmsnorm: input_tensor must use TILE or ROW_MAJOR layout, got {}",
        input.layout());

    const auto& in_shape = input.logical_shape();
    TT_FATAL(
        in_shape.rank() == 4 && in_shape[0] == 1 && in_shape[1] == 1 && in_shape[2] == 1 && in_shape[3] > 0,
        "sharded_rmsnorm: input_tensor must have logical shape [1, 1, 1, D], got {}",
        in_shape);
    const uint32_t d = static_cast<uint32_t>(in_shape[3]);

    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            input.memory_config().buffer_type() == BufferType::L1 && input.shard_spec().has_value(),
        "sharded_rmsnorm: input_tensor must be L1 WIDTH_SHARDED, got {}",
        input.memory_config());
    const auto& in_shard = input.shard_spec().value();
    TT_FATAL(
        in_shard.orientation == ShardOrientation::ROW_MAJOR,
        "sharded_rmsnorm: input_tensor shard orientation must be ROW_MAJOR");
    const uint32_t num_cores = in_shard.grid.num_cores();
    const uint32_t shard_w = in_shard.shape[1];
    TT_FATAL(num_cores >= 1, "sharded_rmsnorm: input_tensor must span at least 1 core");
    TT_FATAL(
        shard_w > 0 && shard_w % TILE_WIDTH == 0,
        "sharded_rmsnorm: input shard width {} must be a positive multiple of {}",
        shard_w,
        TILE_WIDTH);
    TT_FATAL(
        num_cores * shard_w == d, "sharded_rmsnorm: {} cores x shard width {} must equal D={}", num_cores, shard_w, d);

    if (input.layout() == Layout::TILE) {
        const auto& tile = input.tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == TILE_HEIGHT && tile.get_width() == TILE_WIDTH,
            "sharded_rmsnorm: TILE input_tensor must use 32x32 tiles, got {}x{}",
            tile.get_height(),
            tile.get_width());
        TT_FATAL(
            in_shard.shape[0] == TILE_HEIGHT,
            "sharded_rmsnorm: TILE input shard height must be {}, got {}",
            TILE_HEIGHT,
            in_shard.shape[0]);
        TT_FATAL(
            input.padded_shape() == ttnn::Shape({1, 1, TILE_HEIGHT, d}),
            "sharded_rmsnorm: TILE input padded shape must be [1, 1, 32, {}], got {}",
            d,
            input.padded_shape());
    } else {
        TT_FATAL(
            in_shard.shape[0] == 1,
            "sharded_rmsnorm: ROW_MAJOR input shard height must be 1, got {}",
            in_shard.shape[0]);
        TT_FATAL(
            input.padded_shape() == ttnn::Shape({1, 1, 1, d}),
            "sharded_rmsnorm: ROW_MAJOR input padded shape must be [1, 1, 1, {}], got {}",
            d,
            input.padded_shape());
    }

    TT_FATAL(weight.storage_type() == StorageType::DEVICE, "sharded_rmsnorm: weight must be on device");
    TT_FATAL(weight.buffer() != nullptr, "sharded_rmsnorm: weight must be allocated on device");
    TT_FATAL(weight.device() == input.device(), "sharded_rmsnorm: weight must reside on the same device as input");
    TT_FATAL(weight.dtype() == DataType::BFLOAT16, "sharded_rmsnorm: weight must be BFLOAT16");
    TT_FATAL(weight.layout() == Layout::TILE, "sharded_rmsnorm: weight must use TILE layout");
    TT_FATAL(
        weight.tensor_spec().tile().get_height() == TILE_HEIGHT &&
            weight.tensor_spec().tile().get_width() == TILE_WIDTH,
        "sharded_rmsnorm: weight must use 32x32 tiles");
    TT_FATAL(
        weight.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "sharded_rmsnorm: weight must be INTERLEAVED, got {}",
        weight.memory_config().memory_layout());
    TT_FATAL(
        weight.logical_shape() == ttnn::Shape({1, 1, 1, d}),
        "sharded_rmsnorm: weight logical shape must be [1, 1, 1, {}], got {}",
        d,
        weight.logical_shape());

    TT_FATAL(
        attributes.output_layout == Layout::TILE || attributes.output_layout == Layout::ROW_MAJOR,
        "sharded_rmsnorm: output_layout must be TILE or ROW_MAJOR, got {}",
        attributes.output_layout);

    const auto& out_mem = attributes.output_mem_config;
    if (out_mem.memory_layout() == TensorMemoryLayout::INTERLEAVED) {
        TT_FATAL(
            out_mem.buffer_type() == BufferType::DRAM || out_mem.buffer_type() == BufferType::L1,
            "sharded_rmsnorm: interleaved output must be DRAM or L1");
    } else if (out_mem.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_FATAL(
            out_mem.buffer_type() == BufferType::L1 && out_mem.shard_spec().has_value(),
            "sharded_rmsnorm: WIDTH_SHARDED output must be in L1 with a shard spec");
        const auto& out_shard = out_mem.shard_spec().value();
        const uint32_t expected_h = attributes.output_layout == Layout::TILE ? TILE_HEIGHT : 1u;
        TT_FATAL(
            out_shard.grid == in_shard.grid, "sharded_rmsnorm: WIDTH_SHARDED output grid must match input shard grid");
        TT_FATAL(
            out_shard.orientation == ShardOrientation::ROW_MAJOR,
            "sharded_rmsnorm: WIDTH_SHARDED output orientation must be ROW_MAJOR");
        TT_FATAL(
            out_shard.shape[0] == expected_h && out_shard.shape[1] == shard_w,
            "sharded_rmsnorm: WIDTH_SHARDED output shard shape must be [{}, {}], got [{}, {}]",
            expected_h,
            shard_w,
            out_shard.shape[0],
            out_shard.shape[1]);
    } else {
        TT_FATAL(
            false,
            "sharded_rmsnorm: output memory_config must be WIDTH_SHARDED L1 or INTERLEAVED, got {}",
            out_mem.memory_layout());
    }
}

}  // namespace

void ShardedRmsNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_sharded_rmsnorm_tensors(attributes, tensor_args);
}

void ShardedRmsNormDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_sharded_rmsnorm_tensors(attributes, tensor_args);
}

ShardedRmsNormDeviceOperation::spec_return_value_t ShardedRmsNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto page_config = operation_attributes.output_layout == Layout::ROW_MAJOR
                                 ? PageConfig(Layout::ROW_MAJOR)
                                 : PageConfig(Layout::TILE, Tile({TILE_HEIGHT, TILE_WIDTH}));
    return TensorSpec(
        input.logical_shape(), TensorLayout(input.dtype(), page_config, operation_attributes.output_mem_config));
}

ShardedRmsNormDeviceOperation::tensor_return_value_t ShardedRmsNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm

namespace ttnn::prim {

Tensor sharded_rmsnorm(
    const Tensor& input_tensor,
    const Tensor& weight,
    float epsilon,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Layout> output_layout) {
    using OperationType = ttnn::operations::experimental::deepseek::sharded_rmsnorm::ShardedRmsNormDeviceOperation;

    const Layout resolved_layout = output_layout.value_or(input_tensor.layout());
    MemoryConfig resolved_mem_config;
    if (memory_config.has_value()) {
        resolved_mem_config = memory_config.value();
    } else if (input_tensor.shard_spec().has_value()) {
        const auto& in_shard = input_tensor.shard_spec().value();
        const uint32_t out_h = resolved_layout == Layout::TILE ? tt::constants::TILE_HEIGHT : 1u;
        const tt::tt_metal::ShardSpec out_shard(
            in_shard.grid, {out_h, in_shard.shape[1]}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
        resolved_mem_config =
            MemoryConfig(tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, tt::tt_metal::BufferType::L1, out_shard);
    } else {
        resolved_mem_config = input_tensor.memory_config();
    }

    auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon,
        .output_mem_config = std::move(resolved_mem_config),
        .output_layout = resolved_layout,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .weight = weight,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
