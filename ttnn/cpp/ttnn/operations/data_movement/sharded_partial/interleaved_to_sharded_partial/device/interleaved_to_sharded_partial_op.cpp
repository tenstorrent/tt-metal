// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_partial_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <ttnn/operation.hpp>

namespace ttnn::prim {

InterleavedToShardedPartialDeviceOperation::program_factory_t
InterleavedToShardedPartialDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const Tensor& /*input_tensor*/) {
    return InterleavedToShardedPartialProgramFactory{};
}

void InterleavedToShardedPartialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor) {
    const auto& num_slices = operation_attributes.num_slices;
    const auto& slice_index = operation_attributes.slice_index;
    const auto& grid_size = operation_attributes.grid_size;
    const auto& output_dtype = operation_attributes.output_dtype;

    // Validate output tensor
    TT_FATAL(
        slice_index >= 0 && slice_index < num_slices,
        "Slice index and num_slices don't match! Index = {} num_slices = {}",
        slice_index,
        num_slices);
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Currently, only tile layout is supported for partial I->S");
    TT_FATAL(
        (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) % num_slices == 0,
        "Total height of a tensor must be divisible by num_slices!");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must be Interleaved");
    if (input_tensor.dtype() != output_dtype) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "Input tensor layout must be TILE but got {}",
            input_tensor.layout());
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        grid_size.x <= device_grid.x && grid_size.y <= device_grid.y,
        "Grid size for sharding must be less than or equal to total grid available");
}

void InterleavedToShardedPartialDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

TensorSpec InterleavedToShardedPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor) {
    auto shape = input_tensor.padded_shape();

    uint32_t total_height = input_tensor.physical_volume() / shape[-1];
    uint32_t new_height = total_height / operation_attributes.num_slices;

    shape[0] = 1;
    shape[1] = 1;
    shape[2] = new_height;

    auto mem_config = operation_attributes.output_mem_config.with_shard_spec(operation_attributes.shard_spec);

    return TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype, tt::tt_metal::PageConfig(input_tensor.layout()), mem_config));
}

Tensor InterleavedToShardedPartialDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor) {
    auto output_spec = compute_output_specs(operation_attributes, input_tensor);
    return create_device_tensor(output_spec, input_tensor.device());
}

tt::stl::hash::hash_t InterleavedToShardedPartialDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor) {
    auto program_factory = select_program_factory(operation_attributes, input_tensor);
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedPartialDeviceOperation>(
        operation_attributes.grid_size,
        operation_attributes.shard_spec,
        operation_attributes.num_slices,
        operation_attributes.slice_index,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.layout());
}

Tensor interleaved_to_sharded_partial(
    const Tensor& input_tensor,
    const CoreCoord& grid_size,
    const tt::tt_metal::ShardSpec& shard_spec,
    uint32_t num_slices,
    uint32_t slice_index,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype) {
    return ttnn::device_operation::launch<InterleavedToShardedPartialDeviceOperation>(
        InterleavedToShardedPartialParams{
            .grid_size = grid_size,
            .shard_spec = shard_spec,
            .num_slices = num_slices,
            .slice_index = slice_index,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype},
        input_tensor);
}
}  // namespace ttnn::prim
