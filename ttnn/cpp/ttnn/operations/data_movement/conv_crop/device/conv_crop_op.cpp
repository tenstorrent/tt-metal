// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_crop_op.hpp"

#include <magic_enum/magic_enum.hpp>

#include "conv_crop_program_factory.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/buffer_constants.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ConvCropDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    // Rm assert for both, l1 assert for both
    TT_FATAL(input_tensor.memory_config().buffer_type == BufferType::L1, "input tensor must be in L1 buffer");
    TT_FATAL(this->output_mem_config.buffer_type == BufferType::L1, "output tensor must be in L1 buffer");
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "input tensor must be in row major layout");
    TT_FATAL(
        input_tensor.logical_shape()[0] == 1 && input_tensor.logical_shape()[1] == 1,
        "input tensor must be in [1, 1, N * H * W, C] format");
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "input tensor must be height sharded");
    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "output tensor must be height sharded");
    TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "input tensor must be BFLOAT16");

    int post_crop_height = this->pre_crop_height - this->crop_height * 2;  // removes first & last rows
    int post_crop_width = this->pre_crop_width - this->crop_width * 2;     // removes first & last columns

    int total_hw_output_tensor = post_crop_height * post_crop_width;
    TT_FATAL(
        total_hw_output_tensor == this->output_mem_config.shard_spec.value().shape[0] *
                                      this->output_mem_config.shard_spec.value().num_cores(),
        "output tensor shape must match crop size");

    int total_hw_pre_crop = this->pre_crop_height * this->pre_crop_width;
    TT_FATAL(total_hw_pre_crop == input_tensor.get_logical_shape()[2], "input tensor shape must match pre crop size");

    bool same_column_dim =
        input_tensor.memory_config().shard_spec.value().shape[1] == this->output_mem_config.shard_spec.value().shape[1];
    TT_FATAL(same_column_dim, "input and output tensors must have same shard_spec[1] dimension");
    bool full_rows_in_input_shard_shape =
        input_tensor.memory_config().shard_spec.value().shape[0] % pre_crop_width == 0;
    TT_FATAL(full_rows_in_input_shard_shape, "input tensor shard_spec[0] must be multiple of pre_crop_width");
}

std::vector<ttnn::TensorSpec> ConvCropDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    int total_output_hw =
        (this->pre_crop_height - this->crop_height * 2) * (this->pre_crop_width - this->crop_width * 2);

    const Shape output_logical_shape = ttnn::Shape({1, 1, total_output_hw, input_tensor.get_logical_shape()[3]});
    const Shape output_padded_shape = ttnn::Shape({1, 1, total_output_hw, input_tensor.get_padded_shape()[3]});

    return {TensorSpec(
        output_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            output_mem_config,
            output_logical_shape,
            output_padded_shape))};
}

operation::ProgramWithCallbacks ConvCropDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    // each tensor has its respective shard_spec within its memory_config
    return detail::conv_crop_multi_core(
        input_tensor, output_tensor, crop_height, crop_width, pre_crop_height, pre_crop_width);
}
}  // namespace ttnn::operations::data_movement
