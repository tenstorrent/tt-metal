// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_op.hpp"

#include <tt-metalium/constants.hpp>

#include "split_program_factory.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void SplitDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    tt::tt_metal::Buffer* in0_buffer = input_tensor.buffer();

    TT_FATAL(this->dim == 3 || this->dim == 2, "Split is possible along dim 2 or 3 only");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Split does not currently support sharding");
    TT_FATAL(
        this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Split does not currently support sharding");

    TT_FATAL(input_tensor.padded_shape()[0] == 1, "shape[0] must be 1 (batch 1 only)");
    TT_FATAL(
        input_tensor.padded_shape()[this->dim] % this->num_splits == 0,
        "Dim being split must be evenly divisible by number of splits");
    TT_FATAL(
        this->dim <= input_tensor.padded_shape().rank() && this->dim >= 0,
        "Dim being split must be from 0 to rank - 1");
    TT_FATAL(input_tensor.padded_shape().rank() == 4, "Tensor needs to be rank 4");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Tensor needs to be in TILE Layout");
}

std::vector<ttnn::TensorSpec> SplitDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto input_shape_array = input_tensor.padded_shape().to_array_4D();
    auto output_shape_array = input_shape_array;
    output_shape_array[this->dim] /= this->num_splits;
    TensorSpec spec(
        Shape(output_shape_array),
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), output_mem_config));
    return std::vector<ttnn::TensorSpec>(this->num_splits, spec);
}

operation::ProgramWithCallbacks SplitDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return detail::split_last_dim_two_chunks_tiled(input_tensor, output_tensors, this->output_mem_config);
}

}  // namespace ttnn::operations::data_movement
