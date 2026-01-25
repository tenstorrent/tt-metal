// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::bernoulli {

BernoulliDeviceOperation::program_factory_t BernoulliDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void BernoulliDeviceOperation::validate_inputs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Bernoulli: Input tensor need to be on device");
    TT_FATAL(input.buffer() != nullptr, "Bernoulli: Input tensor need to be allocated in buffers on device");
    TT_FATAL((input.layout() == Layout::TILE), "Bernoulli: Input tensor must be tilized");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Bernoulli: Input tensor must be Float32 or Bfloat16");

    if (output.has_value()) {
        TT_FATAL(output.value().storage_type() == StorageType::DEVICE, "Bernoulli: Output tensor need to be on device");
        TT_FATAL(
            output.value().buffer() != nullptr, "Bernoulli: Output tensor need to be allocated in buffers on device");
        TT_FATAL((output.value().layout() == Layout::TILE), "Bernoulli: Output tensor must be tilized");
        TT_FATAL(
            output.value().dtype() == DataType::BFLOAT16 || output.value().dtype() == DataType::FLOAT32,
            "Bernoulli: Output tensor must be Float32 or Bfloat16");
        TT_FATAL(
            input.logical_volume() == output.value().logical_volume(),
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

BernoulliDeviceOperation::spec_return_value_t BernoulliDeviceOperation::compute_output_specs(
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

BernoulliDeviceOperation::tensor_return_value_t BernoulliDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t BernoulliDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto cached_operation_attributes = operation_attributes;
    cached_operation_attributes.seed = 0;
    return tt::stl::hash::hash_objects_with_default_seed(cached_operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::bernoulli

namespace ttnn::prim {
ttnn::operations::bernoulli::BernoulliDeviceOperation::tensor_return_value_t bernoulli(
    const Tensor& input,
    uint32_t seed,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::bernoulli::BernoulliDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        seed,
        dtype.value_or(DataType::FLOAT32),
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input, output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
