// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void CopyDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::BFLOAT8_B or
            input_tensor_a.dtype() == DataType::FLOAT32 or input_tensor_a.dtype() == DataType::BFLOAT4_B or
            input_tensor_a.dtype() == DataType::UINT32 or input_tensor_a.dtype() == DataType::INT32,
        "ttnn.copy only supports float, bfloat and int32 inputs but got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        this->output_dtype == DataType::BFLOAT16 or this->output_dtype == DataType::BFLOAT8_B or
            this->output_dtype == DataType::FLOAT32 or this->output_dtype == DataType::BFLOAT4_B or
            this->output_dtype == DataType::UINT32 or this->output_dtype == DataType::INT32,
        "ttnn.copy only supports float, bfloat and int32 output tensors but got {}",
        this->output_dtype);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy need to be allocated in buffers on device!");

    if (input_tensors.size() == 2) {
        const auto& dst_tensor = input_tensors[1];
        TT_FATAL(input_tensor_a.padded_shape() == dst_tensor.padded_shape(), "Error");
        TT_FATAL(input_tensor_a.layout() == dst_tensor.layout(), "Error");
        TT_FATAL(input_tensor_a.memory_config().memory_layout() == dst_tensor.memory_config().memory_layout(), "Error");
    }
    DataType output_dtype = this->output_dtype;
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        const auto& out_tensor = output_tensors.at(0).value();
        const auto& output_shape_tensor = input_tensors.size() == 2 ? input_tensors[1] : input_tensor_a;
        TT_FATAL(
            out_tensor.logical_shape() == output_shape_tensor.logical_shape() &&
                out_tensor.padded_shape() == output_shape_tensor.padded_shape(),
            "The input tensors need a shape of {}/{}, however the output tensor is only {}/{}",
            output_shape_tensor.logical_shape(),
            output_shape_tensor.padded_shape(),
            out_tensor.logical_shape(),
            out_tensor.padded_shape());
        output_dtype = out_tensor.dtype();
    }
    if (output_dtype != input_tensor_a.dtype()) {
        TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Only tile layout supports dtype conversion");
    }
    auto out_mem_config = (!output_tensors.empty() && output_tensors.at(0).has_value())
                              ? output_tensors.at(0).value().memory_config()
                              : this->output_mem_config;
}

std::vector<ttnn::TensorSpec> CopyDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }
    if (input_tensors.size() == 2) {
        return {input_tensors[1].tensor_spec()};
    }

    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(input_tensor.layout()),
            output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

std::vector<Tensor> CopyDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }
    if (input_tensors.size() == 2) {
        return {input_tensors[1]};
    }
    const auto& input_tensor = input_tensors.at(0);
    auto spec = compute_output_specs(input_tensors, output_tensors)[0];
    return {create_device_tensor(spec, input_tensor.device())};
}

operation::ProgramWithCallbacks CopyDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);

    switch (CopyDeviceOperation::get_parallelization_strategy(input_tensors)) {
        case CopyOpParallelizationStrategy::MULTI_CORE:
        default: return copy_multi_core(input_tensor, output_tensor);
    }
}

CopyOpParallelizationStrategy CopyDeviceOperation::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    return CopyOpParallelizationStrategy::MULTI_CORE;
}

}  // namespace ttnn::operations::data_movement
