// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_op.hpp"

#include "tt_metal/host_api.hpp"

#include "interleaved_to_sharded_program_factory.hpp"


namespace ttnn::operations::data_movement {

void InterleavedToShardedDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(this->output_mem_config.is_sharded(), "Error");
    TT_FATAL(this->output_mem_config.buffer_type == BufferType::L1, "Error");
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL((*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal.get_alignment(HalMemType::L1) == 0, "Shard page size must currently have L1 aligned page size");
    }
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Error");
    }
}


std::vector<tt::tt_metal::LegacyShape> InterleavedToShardedDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}


std::vector<Tensor> InterleavedToShardedDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks InterleavedToShardedDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::interleaved_to_sharded_multi_core(input_tensor, output_tensor);
}


}
