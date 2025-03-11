// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_knit_op.hpp"

#include "conv_knit_program_factory.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/logger.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ConvKnitDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Knit operand needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Knit operand needs to be allocated in buffers on device!");

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Knit operand needs to be height sharded");
    TT_FATAL(input_tensor.memory_config().buffer_type == BufferType::L1, "Knit operand needs to be in L1");
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Knit operand needs to be row major");
    TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "Knit operand needs to be BFLOAT16");

    // Tensor shape needs to be in format
    // [1, 1, N * H * W, C]
    TT_FATAL(
        input_tensor.get_logical_shape().to_array_4D()[0] == 1 &&
            input_tensor.get_logical_shape().to_array_4D()[1] == 1,
        "Knit operand shape needs to be in format: [1, 1, N * H * W, C]");
    // ShardSpec[0] needs to be divisible by input_width
    TT_FATAL(
        input_tensor.shard_spec().value().shape[0] % this->input_width == 0,
        "ShardSpec[0] needs to be divisible by input_width");
}

std::vector<ttnn::TensorSpec> ConvKnitDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.get_logical_shape();
    const auto& input_padded_shape = input_tensor.get_padded_shape();
    // Todo, think about padded shape and logical shape

    log_info(tt::LogOp, "kernel height: {}, num_output_channels: {}", this->kernel_height, this->num_output_channels);
    log_info(tt::LogOp, "input_logical_shape: {}, input_padded_shape: {}", input_shape, input_padded_shape);
    log_info(
        tt::LogOp,
        "input shard spec shape: {} mode {} grid {} full_thing {}",
        input_tensor.memory_config().shard_spec.value().shape,
        input_tensor.memory_config().shard_spec.value().mode,
        input_tensor.memory_config().shard_spec.value().grid,
        input_tensor.memory_config().shard_spec.value());

    const Shape output_logical_shape = ttnn::Shape(
        {1, 1, input_shape.to_array_4D()[2] * this->kernel_height * this->kernel_height, this->num_output_channels});
    const Shape output_padded_shape = ttnn::Shape(
        {1,
         1,
         input_shape.to_array_4D()[2] * this->kernel_height * this->kernel_height,
         tt::round_up(this->num_output_channels, tt::constants::TILE_WIDTH)});  // ?

    log_info(tt::LogOp, "output_logical_shape: {}, output_padded_shape: {}", output_logical_shape, output_padded_shape);
    MemoryConfig output_mem_config = create_sharded_memory_config(
        /*output_logical_shape*/ output_padded_shape,
        input_tensor.memory_config().shard_spec.value().grid,
        ShardStrategy::HEIGHT,
        input_tensor.memory_config().shard_spec.value().orientation,
        std::nullopt,
        input_tensor.get_layout());

    log_info(tt::LogOp, "in cpp: output shard spec shape: {}", output_mem_config.shard_spec.value());

    return {TensorSpec(
        output_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),  // temp, fix it
            PageConfig(input_tensor.get_layout()),
            output_mem_config,  // temp fix it
            output_logical_shape,
            output_padded_shape))};
}

operation::ProgramWithCallbacks ConvKnitDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::conv_knit_multi_core(
        input_tensor,
        output_tensor,
        this->kernel_height,
        this->num_output_channels,
        this->input_width,
        this->num_input_channels);
}

}  // namespace ttnn::operations::data_movement
