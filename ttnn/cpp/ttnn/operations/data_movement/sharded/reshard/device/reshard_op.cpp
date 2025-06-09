// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_op.hpp"

#include <magic_enum/magic_enum.hpp>

#include "reshard_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ReshardDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");

    bool has_output_tensor = output_tensors.size() == 1 && output_tensors[0].has_value();
    if (has_output_tensor) {
        const auto& output_tensor = output_tensors[0].value();
        TT_FATAL(input_tensor.logical_shape() == output_tensor.logical_shape(), "Error");
        TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Error");
        TT_FATAL(input_tensor.layout() == output_tensor.layout(), "Error");
    }
    const auto& out_mem_config =
        has_output_tensor ? output_tensors[0].value().memory_config() : this->output_mem_config;
    TT_FATAL(out_mem_config.is_sharded(), "output must be sharded");

    if ((input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
         out_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED)) {
        TT_FATAL(
            (input_tensor.memory_config().buffer_type() == BufferType::L1 ||
             out_mem_config.buffer_type() == BufferType::L1),
            "Resharding height shard to height shard must have at least one buffer in L1");
    } else if ((input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                out_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED)) {
        TT_FATAL(
            (input_tensor.memory_config().buffer_type() == BufferType::L1 ||
             out_mem_config.buffer_type() == BufferType::L1),
            "Resharding width shard to width shard must have at least one buffer in L1");
    } else {
        TT_FATAL(out_mem_config.buffer_type() == BufferType::L1, "Resharding requires output buffer to be in L1");
    }

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            bool same_row_size = input_tensor.memory_config().shard_spec().value().shape[0] ==
                                 out_mem_config.shard_spec().value().shape[0];
            TT_FATAL(same_row_size, "row major must have shard_spec[0] be the same on both input and output");
        } else {
            bool same_height_size = input_tensor.memory_config().shard_spec().value().shape[1] ==
                                    out_mem_config.shard_spec().value().shape[1];
            TT_FATAL(same_height_size, "row major must have shard_spec[1] be the same on both input and output");
        }
    }
}

std::vector<ttnn::TensorSpec> ReshardDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }

    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.layout(),
            output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

operation::ProgramWithCallbacks ReshardDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    // each tensor has its respective shard_spec within its memory_config
    return detail::reshard_multi_core(input_tensor, output_tensor);
}

std::vector<Tensor> ReshardDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors.at(0).device())};
}

}  // namespace ttnn::operations::data_movement
