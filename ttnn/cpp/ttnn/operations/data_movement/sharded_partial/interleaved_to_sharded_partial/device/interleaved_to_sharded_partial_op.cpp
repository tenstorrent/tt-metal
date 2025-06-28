// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_partial_op.hpp"

#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void InterleavedToShardedPartialDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

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
    if (input_tensor.dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.layout() == Layout::TILE, "Error");
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        this->grid_size.x <= device_grid.x && this->grid_size.y <= device_grid.y,
        "Grid size for sharding must be less than or equal to total grid available");
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<ttnn::TensorSpec> InterleavedToShardedPartialDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shape = input_tensor.padded_shape();

    uint32_t total_height = input_tensor.physical_volume() / shape[-1];
    uint32_t new_height = total_height / this->num_slices;

    shape[0] = 1;
    shape[1] = 1;
    shape[2] = new_height;

    auto mem_config = this->output_mem_config.with_shard_spec(this->shard_spec);

    return {TensorSpec(shape, TensorLayout(output_dtype, PageConfig(input_tensor.layout()), mem_config))};
}

operation::ProgramWithCallbacks InterleavedToShardedPartialDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    // Will move with sharded ops
    return detail::interleaved_to_sharded_multi_core(
        input_tensor, output_tensor, false, this->num_slices, this->slice_index);
}

}  // namespace ttnn::operations::data_movement
