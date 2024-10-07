// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform_device_operation.hpp"

namespace ttnn::operations::uniform {

UniformDeviceOperation::program_factory_t UniformDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return Factory{};
}

void UniformDeviceOperation::validate_inputs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void UniformDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

void UniformDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

UniformDeviceOperation::shape_return_value_t UniformDeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor.get_shape();
}

UniformDeviceOperation::tensor_return_value_t UniformDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input_tensor;
}

std::tuple<UniformDeviceOperation::operation_attributes_t, UniformDeviceOperation::tensor_args_t>
UniformDeviceOperation::invoke(
    const Tensor& input_tensor,
    const int32_t from,
    const int32_t to,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return {
        operation_attributes_t{
            from,
            to,
            memory_config.value_or(input_tensor.memory_config()),
            init_device_compute_kernel_config(
                input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::uniform
