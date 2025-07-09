// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_op.hpp"
#include "tilize_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {
void Tilize::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");

    TT_FATAL(input_tensor_a.physical_volume() % tt::constants::TILE_HW == 0, "Error");

    auto width = input_tensor_a.padded_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = input_tensor_a.physical_volume() / width;
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::FLOAT32 or
            input_tensor_a.dtype() == DataType::UINT32 or input_tensor_a.dtype() == DataType::INT32,
        "data type must be bfloat16, float32, uint32 or int32");

    uint32_t stick_size = stick_s * input_tensor_a.element_size();  // Assuming bfloat16 dataformat

    TT_FATAL((stick_size % 2) == 0, "Stick size must be divisible by 2");

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
        TT_FATAL(this->output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(), "Error");
        TT_FATAL(this->use_multicore == true, "Error");
        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Error");
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
    }
}

std::vector<ttnn::TensorSpec> Tilize::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.memory_config().is_sharded()) {
        auto mem_config = this->output_mem_config.with_shard_spec(input_tensor.memory_config().shard_spec());
        return {TensorSpec(
            input_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::TILE),
                mem_config,
                input_tensor.logical_shape(),
                input_tensor.padded_shape()))};
    }

    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::TILE),
            output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

operation::ProgramWithCallbacks Tilize::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    if (input_tensor_a.memory_config().is_sharded()) {
        return detail::tilize_multi_core_sharded(input_tensor_a, output_tensor);
    }
    if (!this->enough_space_height) {
        return detail::tilize_multi_core_block(input_tensor_a, output_tensor);
    }
    if (!this->use_multicore) {
        return detail::tilize_single_core(input_tensor_a, output_tensor);
    }

    return detail::tilize_multi_core_interleaved(input_tensor_a, output_tensor);
}

}  // namespace ttnn::operations::data_movement
