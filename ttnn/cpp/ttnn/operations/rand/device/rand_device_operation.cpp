// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_device_operation.hpp"
#include <memory>

namespace ttnn::operations::rand {

RandDeviceOperation::program_factory_t RandDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void RandDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(tensor_args.output.storage_type() == StorageType::DEVICE, "Random: Tensor need to be on device");
    TT_FATAL(tensor_args.output.buffer() != nullptr, "Random: Tensor need to be allocated in buffers ondevice");
    TT_FATAL(operation_attributes.from < operation_attributes.to, "Rand: `from` argument must be < `to` argument");
}

void RandDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void RandDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

TensorSpec RandDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ttnn::TensorSpec(
        operation_attributes.shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
}

RandDeviceOperation::tensor_return_value_t RandDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.output;
}

tt::stl::hash::hash_t RandDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto cached_operation_attributes = operation_attributes;
    cached_operation_attributes.seed = 0;
    return tt::stl::hash::hash_objects_with_default_seed(cached_operation_attributes, tensor_args);
}

std::tuple<RandDeviceOperation::operation_attributes_t, RandDeviceOperation::tensor_args_t> RandDeviceOperation::invoke(
    const ttnn::Shape& shape,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    const float from,
    const float to,
    const uint32_t seed) {
    auto output_tensor = create_device_tensor(
        ttnn::TensorSpec(shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
        std::addressof(device));

    return {
        operation_attributes_t{shape, dtype, layout, memory_config, std::addressof(device), from, to, seed},
        tensor_args_t{std::move(output_tensor)}};
}

}  // namespace ttnn::operations::rand
