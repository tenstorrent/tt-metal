// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_for_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

namespace {

tt::tt_metal::TensorSpec make_output_spec(
    const Tensor& input_tensor, const tt::tt_metal::CoreRangeSet& output_core_range_set) {
    const auto& logical_shape = input_tensor.logical_shape();
    const uint32_t logical_height = logical_shape[-2];
    const uint32_t logical_width = logical_shape[-1];
    const uint32_t num_output_cores = output_core_range_set.num_cores();
    const DataType output_dtype =
        input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    auto output_logical_shape = logical_shape;
    output_logical_shape[-2] *= num_output_cores;

    const ShardSpec output_shard_spec{
        output_core_range_set, {logical_height, logical_width}, ShardOrientation::ROW_MAJOR};
    const MemoryConfig output_memory_config{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, output_shard_spec};

    return tt::tt_metal::TensorSpec(
        output_logical_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), output_memory_config));
}

}  // namespace

void AllGatherForMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input must be allocated on device");
    TT_FATAL(input_tensor.is_sharded(), "Input must be sharded");
    const auto input_layout = input_tensor.memory_config().memory_layout();
    const bool width_sharded = input_layout == TensorMemoryLayout::WIDTH_SHARDED;
    const bool height_sharded = input_layout == TensorMemoryLayout::HEIGHT_SHARDED;
    TT_FATAL(width_sharded || height_sharded, "Input must be WIDTH_SHARDED or HEIGHT_SHARDED, got {}", input_layout);
    TT_FATAL(input_tensor.memory_config().buffer_type() == BufferType::L1, "Input must be in L1");
    TT_FATAL(input_tensor.shard_spec().has_value(), "Input must have a shard spec");
    if (width_sharded) {
        TT_FATAL(input_tensor.layout() == Layout::TILE, "WIDTH_SHARDED input must use TILE layout");
    } else {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE || input_tensor.layout() == Layout::ROW_MAJOR,
            "HEIGHT_SHARDED input must use TILE or ROW_MAJOR layout");
        TT_FATAL(
            input_tensor.dtype() != DataType::BFLOAT8_B || input_tensor.layout() == Layout::TILE,
            "ROW_MAJOR HEIGHT_SHARDED input cannot be BFLOAT8_B");
    }
    TT_FATAL(input_tensor.logical_shape().rank() >= 2, "Input rank must be at least 2");
    TT_FATAL(!args.output_core_range_set.empty(), "Output core range set must not be empty");

    const auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    const auto output_bbox = args.output_core_range_set.bounding_box();
    TT_FATAL(
        output_bbox.end_coord.x < device_grid.x && output_bbox.end_coord.y < device_grid.y,
        "Output core range set bounding box {} exceeds device grid {}",
        output_bbox,
        device_grid);
    const auto input_bbox = input_tensor.shard_spec()->grid.bounding_box();
    TT_FATAL(
        input_bbox.end_coord.x < device_grid.x && input_bbox.end_coord.y < device_grid.y,
        "Input shard grid bounding box {} exceeds device grid {}",
        input_bbox,
        device_grid);

    const auto& logical_shape = input_tensor.logical_shape();
    const auto& padded_shape = input_tensor.padded_shape();
    const auto& input_shard = input_tensor.shard_spec().value();
    const uint32_t logical_width = logical_shape[-1];
    const uint32_t padded_width = padded_shape[-1];
    const uint32_t flattened_height = padded_shape.volume() / padded_width;
    TT_FATAL(
        input_shard.shape[0] == flattened_height,
        "Input shard height {} must cover the full flattened tensor height {}",
        input_shard.shape[0],
        flattened_height);
    if (height_sharded) {
        TT_FATAL(
            input_shard.grid.num_cores() == 1,
            "HEIGHT_SHARDED input must reside on a single core, got {} cores",
            input_shard.grid.num_cores());
        TT_FATAL(
            input_shard.shape[1] == padded_width,
            "HEIGHT_SHARDED input shard width {} must equal padded width {}",
            input_shard.shape[1],
            padded_width);
        TT_FATAL(
            padded_width == logical_width,
            "HEIGHT_SHARDED input padded width {} must equal logical width {} so the replica can be multicast as one "
            "contiguous blob",
            padded_width,
            logical_width);
    } else {
        TT_FATAL(
            input_shard.shape[1] * input_shard.grid.num_cores() == padded_width,
            "Input shards must exactly cover padded width {}: {} cores x shard width {}",
            padded_width,
            input_shard.grid.num_cores(),
            input_shard.shape[1]);
    }
    if (input_tensor.layout() == Layout::TILE) {
        TT_FATAL(
            input_shard.shape[0] % TILE_HEIGHT == 0 && input_shard.shape[1] % TILE_WIDTH == 0,
            "TILE input shard shape {} must be tile-aligned",
            input_shard.shape);
    }

    if (tensor_args.preallocated_output.has_value()) {
        const auto expected_spec = make_output_spec(input_tensor, args.output_core_range_set);
        TT_FATAL(
            tensor_args.preallocated_output->tensor_spec() == expected_spec,
            "Preallocated output tensor spec must match the inferred ROW_MAJOR HEIGHT_SHARDED output spec");
    }
}

tt::tt_metal::TensorSpec AllGatherForMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    return make_output_spec(tensor_args.input, args.output_core_range_set);
}

Tensor AllGatherForMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

Tensor all_gather_for_matmul(
    const Tensor& input_tensor,
    const tt::tt_metal::CoreRangeSet& output_core_range_set,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<AllGatherForMatmulDeviceOperation>(
        AllGatherForMatmulParams{.output_core_range_set = output_core_range_set},
        AllGatherForMatmulInputs{.input = input_tensor, .preallocated_output = preallocated_output});
}

}  // namespace ttnn::prim
