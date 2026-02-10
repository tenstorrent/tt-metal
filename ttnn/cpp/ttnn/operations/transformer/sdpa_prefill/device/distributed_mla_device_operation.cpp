// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {

DistributedMLADeviceOperation::program_factory_t DistributedMLADeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return DistributedMlaMeshWorkloadFactory{};
}

void DistributedMLADeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);

    auto input_tensor = tensor_args.input_tensor;

    // Basic validations
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffer on device!");
    TT_FATAL(input_tensor.logical_shape().rank() >= 2, "DistributedMLA requires tensor of rank 2 or greater");
}

void DistributedMLADeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // For now, no additional validation on cache hit
}

DistributedMLADeviceOperation::spec_return_value_t DistributedMLADeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();

    auto mem_config = operation_attributes.memory_config;
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), mem_config));
}

DistributedMLADeviceOperation::tensor_return_value_t DistributedMLADeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_specs, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::transformer::sdpa_prefill

namespace ttnn::prim {
ttnn::Tensor distributed_mla(
    const ttnn::Tensor& input_tensor, std::optional<uint32_t> cluster_axis, const ttnn::MemoryConfig& memory_config) {
    using OperationType = ttnn::operations::transformer::sdpa_prefill::DistributedMLADeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.cluster_axis = cluster_axis, .memory_config = memory_config},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim
