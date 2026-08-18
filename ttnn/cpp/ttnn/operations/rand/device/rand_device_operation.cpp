// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_device_operation.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include <memory>

namespace ttnn::operations::rand {

void RandDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(
        operation_attributes.lower_bound <= operation_attributes.upper_bound,
        "Rand: inclusive lower bound must be <= inclusive upper bound");
}

void RandDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

tt::tt_metal::TensorSpec RandDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    return tt::tt_metal::TensorSpec(
        operation_attributes.shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
}

RandDeviceOperation::tensor_return_value_t RandDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    auto output = create_device_tensor(
        tt::tt_metal::TensorSpec(
            operation_attributes.shape,
            tt::tt_metal::TensorLayout(
                operation_attributes.dtype,
                tt::tt_metal::PageConfig(operation_attributes.layout),
                operation_attributes.memory_config)),
        operation_attributes.device,
        operation_attributes.tensor_topology);
    if (operation_attributes.restricted_mesh_coords.has_value()) {
        // rand has no input tensor from which the operation framework can infer a partial work set.
        output = Tensor(DeviceStorage(output.device_storage(), *operation_attributes.restricted_mesh_coords));
    }
    return output;
}

}  // namespace ttnn::operations::rand

namespace ttnn::prim {
ttnn::operations::rand::RandDeviceOperation::tensor_return_value_t uniform(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config,
    MeshDevice& device,
    float lower_bound,
    float upper_bound,
    uint32_t seed,
    ttsl::SmallVector<bool> mesh_dim_is_sharded,
    std::optional<tt::tt_metal::TensorTopology> tensor_topology) {
    using OperationType = ttnn::operations::rand::RandDeviceOperation;
    std::optional<std::vector<ttnn::MeshCoordinate>> restricted_mesh_coords;
    if (tensor_topology.has_value() && tensor_topology->mesh_coords().size() < device.num_devices()) {
        restricted_mesh_coords = tensor_topology->mesh_coords();
    }
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            shape,
            dtype,
            layout,
            memory_config,
            std::addressof(device),
            lower_bound,
            upper_bound,
            seed,
            std::move(mesh_dim_is_sharded),
            std::move(tensor_topology),
            std::move(restricted_mesh_coords)},
        OperationType::tensor_args_t{});
}
}  // namespace ttnn::prim
