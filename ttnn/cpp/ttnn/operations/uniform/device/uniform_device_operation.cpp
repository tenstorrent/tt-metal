// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform_device_operation.hpp"

namespace ttnn::operations::uniform {

UniformDeviceOperation::program_factory_t UniformDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void UniformDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "Uniform: Input tensor need to be on device");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Uniform: Input tensor need to be allocated in buffers on device");
    TT_FATAL((tensor_args.input.layout() == Layout::TILE), "Uniform: Input tensor must be tilized");
    TT_FATAL(
        tensor_args.input.dtype() == DataType::BFLOAT16 || tensor_args.input.dtype() == DataType::FLOAT32,
        "Uniform: Input tensor must be Float32 or Bfloat16");
    TT_FATAL(operation_attributes.from < operation_attributes.to, "Uniform: from param must be < to");
}

void UniformDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void UniformDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

TensorSpec UniformDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input.tensor_spec();
}

UniformDeviceOperation::tensor_return_value_t UniformDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Since this is an in-place operation, return the input tensor to be updated directly
    return tensor_args.input;
}

tt::stl::hash::hash_t UniformDeviceOperation::compute_program_hash(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto cached_operation_attributes = operation_attributes;
    cached_operation_attributes.seed = 0;
    return tt::stl::hash::hash_objects_with_default_seed(cached_operation_attributes, tensor_args);
}

std::tuple<UniformDeviceOperation::operation_attributes_t, UniformDeviceOperation::tensor_args_t>
UniformDeviceOperation::invoke(
    const Tensor& input,
    const float from,
    const float to,
    const uint32_t seed,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            from,
            to,
            seed,
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        tensor_args_t{input}};
}

}  // namespace ttnn::operations::uniform
