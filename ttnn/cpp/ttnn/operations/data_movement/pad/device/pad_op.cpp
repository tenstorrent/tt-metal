// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "pad_op.hpp"
#include "pad_program_factory.hpp"

namespace ttnn::operations::data_movement {

void Pad::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operand to pad needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE || input_tensor.get_layout() == Layout::ROW_MAJOR, "Error");
    if (input_tensor.get_layout() == Layout::TILE) {
        TT_FATAL(
            (this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 && this->input_tensor_start[3] == 0),
            "On device padding only supports padding at end of dims"
        );
    }
    TT_FATAL(input_tensor.get_shape().with_tile_padding()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_shape().with_tile_padding()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_shape().with_tile_padding()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_shape().with_tile_padding()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3], "Output size cannot fit input with offset");

    if (input_tensor.get_layout() == Layout::TILE) {
        TT_FATAL((this->output_tensor_shape[2] % TILE_HEIGHT == 0), "Can only pad tilized tensor with full tiles");
        TT_FATAL((this->output_tensor_shape[3] % TILE_WIDTH == 0), "Can only pad tilized tensor with full tiles");
        TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16, "Cannot pad tilized tensor with specified format");
    } else if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(this->output_tensor_shape[3] % 2 == 0, "RM padding requires output X dim to be a multiple of 2");
        TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16, "Cannot pad RM tensor with specified format");
    }

    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
        TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Error");

        TT_FATAL(this->output_mem_config.is_sharded(), "Error");
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    }
}

std::vector<ttnn::Shape> Pad::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    return {ttnn::Shape(this->output_tensor_shape)};
}

std::vector<Tensor> Pad::create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto shapes = compute_output_shapes(input_tensors);

    if (this->output_mem_config.is_sharded()) {
        return {create_device_tensor(
            shapes[0],
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            this->output_mem_config)};
    } else {
        return {create_device_tensor(shapes[0], input_tensor.get_dtype(), input_tensor.get_layout(), input_tensor.device(), this->output_mem_config)};
    }
}

operation::ProgramWithCallbacks Pad::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        if (input_tensor.is_sharded()) {
            return detail::pad_rm_sharded(input_tensor, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
        } else {
            if (use_multicore) {
                return detail::pad_rm_reader_writer_multi_core_v2(input_tensor, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
            } else {
                return detail::pad_rm_reader_writer(input_tensor, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
            }
        }
    } else if (input_tensor.get_layout() == Layout::TILE) {
        if (this->use_multicore) {
            tt::log_warning(tt::LogType::LogOp, "TILE layout does not have multicore implementation yet. Falling back to 1 core.");
        }
        return detail::pad_tile(input_tensor, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
    } else {
        TT_THROW("Unsupported layout for pad");
        return {};
    }

}


}  // namespace ttnn::operations::reduction
