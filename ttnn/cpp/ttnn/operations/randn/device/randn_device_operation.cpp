// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "randn_device_operation.hpp"
#include <memory>

namespace ttnn::operations::randn {

RandnDeviceOperation::program_factory_t RandnDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void RandnDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        operation_attributes.dtype == DataType::FLOAT32 || operation_attributes.dtype == DataType::BFLOAT16,
        "Randn: Output tensor must be Float32 or Bfloat16");
    TT_FATAL(operation_attributes.layout == Layout::TILE, "Randn: Not currently supporting row major layout");
}

void RandnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void RandnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

TensorSpec RandnDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ttnn::TensorSpec(
        operation_attributes.shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
}

RandnDeviceOperation::tensor_return_value_t RandnDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        ttnn::TensorSpec(
            operation_attributes.shape,
            tt::tt_metal::TensorLayout(
                operation_attributes.dtype,
                tt::tt_metal::PageConfig(operation_attributes.layout),
                operation_attributes.memory_config)),
        operation_attributes.device);
}

tt::stl::hash::hash_t RandnDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto cached_operation_attributes = operation_attributes;
    cached_operation_attributes.seed = 0;
    return tt::stl::hash::hash_objects_with_default_seed(cached_operation_attributes, tensor_args);
}

std::tuple<RandnDeviceOperation::operation_attributes_t, RandnDeviceOperation::tensor_args_t>
RandnDeviceOperation::invoke(
    const ttnn::Shape& shape,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const uint32_t seed) {
    return {
        operation_attributes_t{
            shape,
            dtype,
            layout,
            memory_config,
            std::addressof(device),
            init_device_compute_kernel_config(
                device.arch(), compute_kernel_config, MathFidelity::HiFi4, false, true, false, true),
            seed,
        },
        tensor_args_t{}};
}

}  // namespace ttnn::operations::randn
