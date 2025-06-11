// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved_op.hpp"

#include "sharded_to_interleaved_program_factory.hpp"
#include <tt-metalium/hal.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ShardedToInterleavedDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shard_spec = input_tensor.shard_spec().value();
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.memory_config().is_sharded(), "Input tensor must be sharded");
    TT_FATAL(input_tensor.memory_config().buffer_type() == BufferType::L1, "Input tensor must be in L1");
    TT_FATAL(
        this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output memory config must be Interleaved");
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        uint32_t l1_alignment = hal::get_l1_alignment();
        TT_FATAL(
            (*input_tensor.memory_config().shard_spec()).shape[1] * input_tensor.element_size() % (l1_alignment) == 0,
            "Shard page size must be aligned to {}B for L1 Tensor",
            l1_alignment);
    }
    if (input_tensor.dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.layout() == Layout::TILE, "If diff output type, tensor must be TILED");
    }
}

std::vector<ttnn::TensorSpec> ShardedToInterleavedDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(input_tensor.layout()),
            output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

operation::ProgramWithCallbacks ShardedToInterleavedDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::sharded_to_interleaved_multi_core(input_tensor, output_tensor, this->is_l1_aligned);
}

}  // namespace ttnn::operations::data_movement
