// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_op.hpp"

#include "tt_metal/host_api.hpp"

#include "sharded_to_interleaved_program_factory.hpp"

namespace ttnn::operations::data_movement {

void ShardedToInterleavedDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {

    const auto& input_tensor = input_tensors.at(0);
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.memory_config().is_sharded(), "Input tensor must be sharded");
    TT_FATAL(input_tensor.memory_config().buffer_type == BufferType::L1, "Input tensor must be in L1");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Output memory config must be Interleaved");
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL((*input_tensor.memory_config().shard_spec).shape[1] * input_tensor.element_size() % (this->output_mem_config.buffer_type == BufferType::DRAM ? DRAM_ALIGNMENT : L1_ALIGNMENT) == 0, "Shard page size must be aligned to 16B for L1 Tensor, or 32B for DRAM tensor");
    }
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE, "If diff output type, tensor must be TILED");
    }
}


std::vector<ttnn::Shape> ShardedToInterleavedDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_shape().with_tile_padding()};
}


std::vector<Tensor> ShardedToInterleavedDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks ShardedToInterleavedDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::sharded_to_interleaved_multi_core(input_tensor, output_tensor);
}


}
