// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_op.hpp"
#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_utils.hpp"
#include "reshape_program_factory.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ReshapeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::FLOAT32, "Error");

    TT_FATAL(
        input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR,
        "Only tile and row major reshape supported!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Use ttnn::reshape for reshaping sharded inputs");
    TT_FATAL(
        this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Reshape does not currently support sharding. Use ttnn::reshape for reshaping sharded inputs");

    if (input_tensor_a.layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.physical_volume() % TILE_HW == 0, "Error");
    } else if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        uint32_t ROW_MAJOR_WIDTH = 8;
        TT_FATAL(
            input_tensor_a.padded_shape()[3] % ROW_MAJOR_WIDTH == 0 && padded_output_shape[3] % ROW_MAJOR_WIDTH == 0,
            "Operand/target width must be a multiple of 8");
        uint32_t num_old_sticks =
            input_tensor_a.padded_shape()[0] * input_tensor_a.padded_shape()[1] * input_tensor_a.padded_shape()[2];
        uint32_t num_new_sticks = padded_output_shape[0] * padded_output_shape[1] * padded_output_shape[2];
    } else {
        TT_THROW("Unsupported layout for reshape");
    }
}

std::vector<ttnn::TensorSpec> ReshapeDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        logical_output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            output_mem_config,
            logical_output_shape,
            padded_output_shape))};
}

operation::ProgramWithCallbacks ReshapeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        return {detail::reshape_rm_multi_core(input_tensor_a, output_tensor)};
    } else if (input_tensor_a.layout() == Layout::TILE) {
        return {detail::reshape_tile_single_core(input_tensor_a, output_tensor)};
    } else {
        TT_ASSERT(false, "Unsupported layout for reshape");
        return {};
    }
}

}  // namespace ttnn::operations::data_movement
