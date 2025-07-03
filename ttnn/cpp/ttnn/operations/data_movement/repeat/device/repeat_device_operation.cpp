// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/data_movement/repeat/device/host/repeat_program_factory.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_device_operation.hpp"

namespace ttnn {

void RepeatDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    // Validate the input tensor
    const Tensor& input_tensor_a = input_tensors.at(0);
    TT_FATAL(
        input_tensor_a.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "This function is for RM->RM");
    TT_FATAL(
        input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 or
            input_tensor_a.dtype() == tt::tt_metal::DataType::UINT32 or
            input_tensor_a.dtype() == tt::tt_metal::DataType::FLOAT32,
        "Can only work with bfloat16/float32 or uint32 tensors");
    // is this relevant?
    TT_FATAL(
        this->m_output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(),
        "Output tensor must have the same memory layout as input tensor");
}

std::vector<TensorSpec> RepeatDeviceOperation::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto output_shape = input_tensor_a.logical_shape();
    output_shape[m_is_last_dim ? -1 : 1] *= m_num_repeats;

    auto mem_config = this->m_output_mem_config;
    if (input_tensor_a.memory_config().is_sharded()) {
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = output_shape[0];
        mem_config = mem_config.with_shard_spec(shard_spec);
    }
    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor_a.dtype(), tt::tt_metal::PageConfig(input_tensor_a.layout()), mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks RepeatDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    return operations::data_movement::repeat::rm_repeat_program_factory(
        input_tensors.at(0), m_num_repeats, output_tensors.at(0), m_is_last_dim);
}
}  // namespace ttnn
