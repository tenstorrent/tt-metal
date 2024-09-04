// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_op.hpp"
#include "tilize_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/constants.hpp"
namespace ttnn::operations::data_movement {
void Tilize::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Can only tilize row major data");

    TT_FATAL(input_tensor_a.volume() % tt::constants::TILE_HW == 0);

    auto width = input_tensor_a.get_legacy_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = input_tensor_a.volume() / width;
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16);

    uint32_t stick_size = stick_s * input_tensor_a.element_size();  // Assuming bfloat16 dataformat

    TT_FATAL((stick_size % 2) == 0, "Stick size must be divisible by 2");

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(this->output_mem_config.memory_layout == input_tensor_a.memory_config().memory_layout);
        TT_FATAL(this->use_multicore == true);
        TT_FATAL(input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR);
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
    }
}

std::vector<tt::tt_metal::Shape> Tilize::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto output_shape = input_tensor_a.get_legacy_shape();
    return {output_shape};
}

std::vector<Tensor> Tilize::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.memory_config().is_sharded()) {
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = input_tensor.memory_config().shard_spec;
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            this->output_dtype,
            Layout::TILE,
            input_tensor.device(),
            mem_config)};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Tilize::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (this->use_multicore) {
        return detail::tilize_multi_core(input_tensor_a, output_tensor);
    }
    return detail::tilize_single_core(input_tensor_a, output_tensor);
}

}  // namespace ttnn::operations::data_movement
