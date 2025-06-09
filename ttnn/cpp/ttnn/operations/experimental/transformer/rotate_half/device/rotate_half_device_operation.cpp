// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_device_operation.hpp"

#include "single_core/rotate_half_program_factory.hpp"

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt::tt_metal;

void RotateHalf::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to rotate half need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to rotate half need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to rotate half must be tilized");
    TT_FATAL(input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "RotateHalf does not currently support sharding");
    TT_FATAL(
        this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "RotateHalf does not currently support sharding");
}

std::vector<ttnn::TensorSpec> RotateHalf::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(Layout::TILE),
            output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

operation::ProgramWithCallbacks RotateHalf::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return detail::rotate_half_single_core(input_tensor, output_tensor);
}

RotateHalfOpParallelizationStrategy RotateHalf::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return RotateHalfOpParallelizationStrategy::SINGLE_CORE;
}

}  // namespace ttnn::operations::experimental::transformer
