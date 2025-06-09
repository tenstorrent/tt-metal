// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/host/reshape_program_factory.hpp"

#include <cstdint>

using namespace tt::tt_metal;

namespace ttnn {

void ReshapeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    // Validate the input tensor
    const Tensor& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::UINT32 or
            input_tensor_a.dtype() == DataType::FLOAT32,
        "Can only work with bfloat16/float32 or uint32 tensors");
    TT_FATAL(
        this->output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(),
        "Output tensor must have the same memory layout as input tensor");
}

std::vector<TensorSpec> ReshapeDeviceOperation::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto mem_config = this->output_mem_config;
    if (input_tensor_a.memory_config().is_sharded()) {
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = logical_output_shape[0];
        mem_config = mem_config.with_shard_spec(shard_spec);
    }
    return {TensorSpec(
        logical_output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor_a.dtype(),
            PageConfig(input_tensor_a.layout()),
            mem_config,
            logical_output_shape,
            padded_output_shape))};
}

operation::ProgramWithCallbacks ReshapeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    if (input_tensors.at(0).layout() == Layout::ROW_MAJOR) {
        return operations::data_movement::reshape::rm_reshape_preparer(input_tensors.at(0), output_tensors.at(0));
    } else {
        return operations::data_movement::reshape::reshape_tiled_program_factory(
            input_tensors.at(0), output_tensors.at(0));
    }
}
}  // namespace ttnn
