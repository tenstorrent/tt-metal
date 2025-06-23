// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_partial_op.hpp"

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ShardedToInterleavedPartialDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shard_spec = input_tensor.shard_spec().value();

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

    TT_FATAL(input_tensor.memory_config().is_sharded(), "Error");
    if (input_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        if (input_tensor.padded_shape()[-1] % shard_spec.shape[1] != 0 ||
            ((input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) % shard_spec.shape[0]) != 0) {
            TT_FATAL(input_tensor.shard_spec().value().grid.ranges().size() == 1, "Error");
        }
    }
    if (input_tensor.dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.layout() == Layout::TILE, "Error");
    }
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<ttnn::TensorSpec> ShardedToInterleavedPartialDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // Don't create anything, we already passed in output tensor
    return {};
}

operation::ProgramWithCallbacks ShardedToInterleavedPartialDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = input_tensors[1];
    // Will move with sharded ops
    return detail::sharded_to_interleaved_multi_core(
        input_tensor, output_tensor, false, this->num_slices, this->slice_index);
}

}  // namespace ttnn::operations::data_movement
