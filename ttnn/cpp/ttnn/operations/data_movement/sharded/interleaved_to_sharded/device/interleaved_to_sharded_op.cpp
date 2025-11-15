// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>

#include "interleaved_to_sharded_op.hpp"
#include "interleaved_to_sharded_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void InterleavedToShardedDeviceOperation::validate_with_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        const auto& output_tensor = output_tensors[0].value();
        TT_FATAL(output_tensor.logical_shape() == input_tensor.logical_shape(), "Mismatched output shape");
        TT_FATAL(output_tensor.memory_config() == this->output_mem_config, "Mismatched output memory config");
        TT_FATAL(output_tensor.dtype() == this->output_dtype, "Mismatched output dtype");
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
        TT_FATAL(output_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
        TT_FATAL(output_tensor.device() == input_tensor.device(), "Operands to shard need to be on the same device!");
    }

    TT_FATAL(input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Input tensor memory layout must be INTERLEAVED but got {}", input_tensor.memory_config().memory_layout());
    TT_FATAL(this->output_mem_config.is_sharded(), "Output memory config must be sharded");
    if (this->output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(this->output_mem_config.buffer_type() == BufferType::L1, "We don't support DRAM block sharding");
    }
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        TT_FATAL((*this->output_mem_config.shard_spec()).shape[1] * input_tensor.element_size() % hal::get_l1_alignment() == 0, "Shard page size must currently have L1 aligned page size");
    }
    if (input_tensor.dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor layout must be TILE when dtype conversion is needed but got {}", input_tensor.layout());
    }
}


std::vector<ttnn::TensorSpec> InterleavedToShardedDeviceOperation::compute_output_specs(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(input_tensor.logical_shape(), TensorLayout::fromPaddedShape(
        output_dtype,
        PageConfig(input_tensor.layout()),
        output_mem_config,
        input_tensor.logical_shape(),
        input_tensor.padded_shape()))};
}

std::vector<Tensor> InterleavedToShardedDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }
    const auto& input_tensor = input_tensors.at(0);
    auto spec = compute_output_specs(input_tensors, output_tensors)[0];
    return {create_device_tensor(spec, input_tensor.device())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
InterleavedToShardedDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

operation::ProgramWithCallbacks InterleavedToShardedDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::interleaved_to_sharded_multi_core(input_tensor, output_tensor, this->keep_l1_aligned);
}


}
