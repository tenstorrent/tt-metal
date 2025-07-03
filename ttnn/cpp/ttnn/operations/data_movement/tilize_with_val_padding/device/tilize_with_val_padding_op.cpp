// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_op.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "tilize_with_val_padding_program_factory.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void TilizeWithValPadding::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_shape = input_tensor_a.padded_shape();
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::INT32 or
            input_tensor_a.dtype() == DataType::UINT32 or input_tensor_a.dtype() == DataType::FLOAT32,
        "Can only tilize bfloat16/float32 or int32/uint32 tensors");

    TT_FATAL(input_shape.rank() >= 1, "Input tensor must be of rank >= 1, but its shape is {}", input_shape);

    for (auto i = 0; i < input_shape.rank(); i++) {
        TT_FATAL(
            input_shape[i] <= this->output_padded_shape[i],
            "Output tensor shape {} must be greater than or equal to input shape {} in each dimension, but is smaller "
            "in dimension {}",
            this->output_padded_shape,
            input_shape,
            i);
    }

    uint32_t num_rows = this->output_padded_shape[-1];
    uint32_t inner_dim = this->output_padded_shape[-2];
    TT_FATAL(
        inner_dim % TILE_WIDTH == 0 && num_rows % TILE_HEIGHT == 0,
        "To be tilizable output tensor shape {} must be divisible by tile size ({}, {})",
        output_padded_shape,
        TILE_WIDTH,
        TILE_HEIGHT);

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "Input tensor must be width sharded");
        TT_FATAL(
            this->output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(),
            "Output tensor must have the same memory layout as input tensor");
        for (uint32_t i = 0; i < input_tensor_a.padded_shape().rank(); i++) {
            if (i != input_shape.rank() - 2) {
                TT_FATAL(input_shape[i] == this->output_padded_shape[i], "Error");
            }
        }
    }
}

std::vector<ttnn::TensorSpec> TilizeWithValPadding::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto input_shape = input_tensors.at(0).padded_shape();

    if (input_tensor.memory_config().is_sharded()) {
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] = output_padded_shape.volume() / output_padded_shape[-1];
        auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
        return {TensorSpec(
            input_shape,
            TensorLayout::fromPaddedShape(
                output_dtype, PageConfig(Layout::TILE), mem_config, input_shape, output_padded_shape))};
    }

    return {TensorSpec(
        input_shape,
        TensorLayout::fromPaddedShape(
            output_dtype, PageConfig(Layout::TILE), output_mem_config, input_shape, output_padded_shape))};
}

// TODO: If pad is called on a tile and output is not tile, we could untilize then pad, and output is RM
// Currently calling pad on a tile requires the output pad shape to be tile
operation::ProgramWithCallbacks TilizeWithValPadding::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor_a.memory_config().is_sharded()) {
        return detail::tilize_with_val_padding_multi_core_sharded(input_tensor_a, output_tensor, this->pad_value);
    }
    if (!this->enough_space_height) {
        return detail::tilize_with_val_padding_multi_core_block_interleaved(
            input_tensor_a, output_tensor, this->pad_value);
    }
    if (!this->use_multicore) {
        return detail::tilize_with_val_padding_single_core(input_tensor_a, output_tensor, this->pad_value);
    }

    return detail::tilize_with_val_padding_multi_core_interleaved(input_tensor_a, output_tensor, this->pad_value);
}

}  // namespace ttnn::operations::data_movement
