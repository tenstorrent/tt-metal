// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "pad_op.hpp"
#include "pad_program_factory.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::data_movement {

void Pad::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    auto logical_rank = input_tensor.logical_shape().rank();
    auto padded_rank = input_tensor.padded_shape().rank();
    TT_FATAL(logical_rank == padded_rank, "ttnn.pad: logical and padded shapes must have the same rank");
    TT_FATAL(input_tensor.logical_shape().rank() <= 4, "ttnn.pad: input tensor rank currently must be 4 or less");
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operand to pad needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_FATAL(
        input_tensor.get_layout() == tt::tt_metal::Layout::TILE ||
            input_tensor.get_layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Error");
    if (input_tensor.get_layout() == Layout::TILE) {
        TT_FATAL(
            (this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 &&
             this->input_tensor_start[3] == 0),
            "On device padding only supports padding at end of dims");
    }
    TT_FATAL(
        input_tensor.get_padded_shape()[0] + this->input_tensor_start[0] <= this->output_padded_shape[0],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.get_padded_shape()[1] + this->input_tensor_start[1] <= this->output_padded_shape[1],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.get_padded_shape()[2] + this->input_tensor_start[2] <= this->output_padded_shape[2],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.get_padded_shape()[3] + this->input_tensor_start[3] <= this->output_padded_shape[3],
        "Output size cannot fit input with offset");

    if (input_tensor.get_layout() == Layout::TILE) {
        TT_FATAL((this->output_padded_shape[2] % TILE_HEIGHT == 0), "Can only pad tilized tensor with full tiles");
        TT_FATAL((this->output_padded_shape[3] % TILE_WIDTH == 0), "Can only pad tilized tensor with full tiles");
        TT_FATAL(
            input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16,
            "Cannot pad tilized tensor with specified format");
    } else if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16,
            "Cannot pad RM tensor with specified format");
    }

    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "ttnn.pad: For sharded inputs, only height-sharding is supported.");
        TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "ttnn.pad: Only row-major sharded inputs are supported.");

        TT_FATAL(this->output_mem_config.is_sharded(), "ttnn.pad: For sharded inputs, the output must be sharded.");
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "ttnn.pad: for sharded inputs, only height-sharding is supported for the output.");
    }
}

std::vector<ttnn::TensorSpec> Pad::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        output_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.get_dtype(),
            PageConfig(input_tensor.get_layout()),
            output_mem_config,
            output_logical_shape,
            output_padded_shape))};
}

operation::ProgramWithCallbacks Pad::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        if (input_tensor.is_sharded()) {
            uint32_t input_tot_h = std::accumulate(
                input_tensor.get_logical_shape().view().begin(),
                input_tensor.get_logical_shape().view().end() - 1,
                1,
                std::multiplies<uint32_t>());
            uint32_t input_w = input_tensor.get_logical_shape()[3];

            uint32_t output_tot_h = std::accumulate(
                output_tensor.get_logical_shape().view().begin(),
                output_tensor.get_logical_shape().view().end() - 1,
                1,
                std::multiplies<uint32_t>());
            uint32_t output_w = output_tensor.get_logical_shape()[3];

            if (input_w != output_w and input_tot_h != output_tot_h) {
                TT_THROW(
                    "ttnn.pad: Unsupported sharded row-major padding configuration: pad_impl did not decompose padding "
                    "correctly.");
                return {};
            } else if (input_w != output_w) {
                return detail::pad_rm_sharded_width_only(
                    input_tensor, output_tensor, this->output_padded_shape, this->input_tensor_start, this->pad_value);
            } else if (input_tot_h != output_tot_h) {
                return detail::pad_rm_sharded_height_only(
                    input_tensor, output_tensor, this->output_padded_shape, this->input_tensor_start, this->pad_value);
            } else {
                // for no padding, we just use the height-only padding program
                return detail::pad_rm_sharded_height_only(
                    input_tensor, output_tensor, this->output_padded_shape, this->input_tensor_start, this->pad_value);
            }
        } else {
            if (use_multicore) {
                return detail::pad_rm_reader_writer_multi_core_v2(
                    input_tensor, output_tensor, this->output_padded_shape, this->input_tensor_start, this->pad_value);
            } else {
                return detail::pad_rm_reader_writer(
                    input_tensor, output_tensor, this->output_padded_shape, this->input_tensor_start, this->pad_value);
            }
        }
    } else if (input_tensor.get_layout() == Layout::TILE) {
        if (this->use_multicore) {
            tt::log_warning(
                tt::LogType::LogOp, "TILE layout does not have multicore implementation yet. Falling back to 1 core.");
        }
        return detail::pad_tile(
            input_tensor, output_tensor, this->output_padded_shape, this->input_tensor_start, this->pad_value);
    } else {
        TT_THROW("Unsupported layout for pad");
        return {};
    }
}

}  // namespace ttnn::operations::data_movement
