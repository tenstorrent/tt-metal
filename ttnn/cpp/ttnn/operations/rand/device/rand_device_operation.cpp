// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include <memory>

namespace ttnn::operations::rand {

RandDeviceOperation::program_factory_t RandDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void RandDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
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
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    return ttnn::TensorSpec(
        operation_attributes.shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
}

RandDeviceOperation::tensor_return_value_t RandDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    return create_device_tensor(
        ttnn::TensorSpec(
            operation_attributes.shape,
            tt::tt_metal::TensorLayout(
                operation_attributes.dtype,
                tt::tt_metal::PageConfig(operation_attributes.layout),
                operation_attributes.memory_config)),
        operation_attributes.device);
}

tt::stl::hash::hash_t RandDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto cached_operation_attributes = operation_attributes;
    cached_operation_attributes.seed = 0;
    return tt::stl::hash::hash_objects_with_default_seed(cached_operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::rand

namespace ttnn::prim {
ttnn::operations::rand::RandDeviceOperation::tensor_return_value_t uniform(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    float from,
    float to,
    uint32_t seed) {
    using OperationType = ttnn::operations::rand::RandDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            shape, dtype, layout, memory_config, std::addressof(device), from, to, seed},
        OperationType::tensor_args_t{});
}
}  // namespace ttnn::prim
