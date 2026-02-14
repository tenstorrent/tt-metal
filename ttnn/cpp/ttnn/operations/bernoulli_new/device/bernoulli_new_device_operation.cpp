// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli_new_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bernoulli_new {

BernoulliNewDeviceOperation::program_factory_t BernoulliNewDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void BernoulliNewDeviceOperation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "BernoulliNew: Input tensor need to be on device");
    TT_FATAL(input.buffer() != nullptr, "BernoulliNew: Input tensor need to be allocated in buffers on device");
    TT_FATAL((input.layout() == Layout::TILE), "BernoulliNew: Input tensor must be tilized");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "BernoulliNew: Input tensor must be Float32 or Bfloat16");

    if (output.has_value()) {
        TT_FATAL(
            output.value().storage_type() == StorageType::DEVICE, "BernoulliNew: Output tensor need to be on device");
        TT_FATAL(
            output.value().buffer() != nullptr,
            "BernoulliNew: Output tensor need to be allocated in buffers on device");
        TT_FATAL((output.value().layout() == Layout::TILE), "BernoulliNew: Output tensor must be tilized");
        TT_FATAL(
            output.value().dtype() == DataType::BFLOAT16 || output.value().dtype() == DataType::FLOAT32,
            "BernoulliNew: Output tensor must be Float32 or Bfloat16");
        TT_FATAL(
            input.logical_volume() == output.value().logical_volume(),
            "BernoulliNew: Output and input tensor shape must be equal");
    }
}

void BernoulliNewDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

BernoulliNewDeviceOperation::spec_return_value_t BernoulliNewDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }

    auto output_shape = tensor_args.input.logical_shape();
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype, tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.memory_config));
}

BernoulliNewDeviceOperation::tensor_return_value_t BernoulliNewDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::bernoulli_new

namespace ttnn::prim {
ttnn::operations::bernoulli_new::BernoulliNewDeviceOperation::tensor_return_value_t bernoulli_new(
    const Tensor& input,
    uint32_t seed,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::bernoulli_new::BernoulliNewDeviceOperation;
    TT_FATAL(input.device() != nullptr, "BernoulliNew: Input tensor needs to be on device");

    auto operation_attributes = OperationType::operation_attributes_t{
        seed,
        dtype.value_or(DataType::FLOAT32),
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input, output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
