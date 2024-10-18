// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli_device_operation.hpp"

namespace ttnn::operations::bernoulli {

BernoulliDeviceOperation::program_factory_t BernoulliDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void BernoulliDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Bernoulli: Input tensor need to be on device");
    TT_FATAL(input.buffer() != nullptr, "Bernoulli: Input tensor need to be allocated in buffers on device");
    TT_FATAL((input.get_layout() == Layout::TILE), "Bernoulli: Input tensor must be tilized");
    TT_FATAL(
        input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::FLOAT32,
        "Bernoulli: Input tensor must be Float32 or Bfloat16");

    if (output.has_value()) {
        TT_FATAL(output.value().storage_type() == StorageType::DEVICE, "Bernoulli: Output tensor need to be on device");
        TT_FATAL(
            output.value().buffer() != nullptr, "Bernoulli: Output tensor need to be allocated in buffers on device");
        TT_FATAL((output.value().get_layout() == Layout::TILE), "Bernoulli: Output tensor must be tilized");
        TT_FATAL(
            output.value().get_dtype() == DataType::BFLOAT16 || output.value().get_dtype() == DataType::FLOAT32,
            "Bernoulli: Output tensor must be Float32 or Bfloat16");
        TT_FATAL(
            input.get_logical_volume() == output.value().get_logical_volume(),
            "Bernoulli: Output and input tensor shape must be equal");
    }
}

void BernoulliDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void BernoulliDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

BernoulliDeviceOperation::shape_return_value_t BernoulliDeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input.get_logical_shape();
}

BernoulliDeviceOperation::tensor_return_value_t BernoulliDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }

    auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    return create_device_tensor(
        output_shapes,
        operation_attributes.dtype,
        Layout::TILE,
        tensor_args.input.device(),
        operation_attributes.memory_config);
}

std::tuple<BernoulliDeviceOperation::operation_attributes_t, BernoulliDeviceOperation::tensor_args_t>
BernoulliDeviceOperation::invoke(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            dtype.value_or(DataType::FLOAT32),
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        tensor_args_t{input, output}};
}

}  // namespace ttnn::operations::bernoulli
