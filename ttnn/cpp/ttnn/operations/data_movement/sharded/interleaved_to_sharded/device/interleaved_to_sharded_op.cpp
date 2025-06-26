// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>

#include "interleaved_to_sharded_op.hpp"
#include "interleaved_to_sharded_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void InterleavedToShardedDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    TT_FATAL(input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(this->output_mem_config.is_sharded(), "Error");
    if (this->output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(this->output_mem_config.buffer_type() == BufferType::L1, "We don't support DRAM block sharding");
    }
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL((*this->output_mem_config.shard_spec()).shape[1] * input_tensor.element_size() % hal::get_l1_alignment() == 0, "Shard page size must currently have L1 aligned page size");
    }
    if (input_tensor.dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.layout() == Layout::TILE, "Error");
    }
}


std::vector<ttnn::TensorSpec> InterleavedToShardedDeviceOperation::compute_output_specs(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(input_tensor.logical_shape(), TensorLayout::fromPaddedShape(
        output_dtype,
        PageConfig(input_tensor.layout()),
        output_mem_config,
        input_tensor.logical_shape(),
        input_tensor.padded_shape()))};
}

operation::ProgramWithCallbacks InterleavedToShardedDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::interleaved_to_sharded_multi_core(input_tensor, output_tensor, this->keep_l1_aligned);
}


}
