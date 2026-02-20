// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full {

FullDeviceOperation::program_factory_t FullDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.memory_config.is_sharded()) {
        if (operation_attributes.memory_config.shard_spec().has_value()) {
            return FullShardedProgramFactory{};
        }
        return FullNDShardedProgramFactory{};
    }
    return FullInterleavedProgramFactory{};
}

void FullDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(
        operation_attributes.dtype == DataType::BFLOAT16 || operation_attributes.dtype == DataType::INT32 ||
            operation_attributes.dtype == DataType::FLOAT32,
        "Full: Unsupported data type {}",
        operation_attributes.dtype);

    const auto shape = operation_attributes.shape;

    TT_FATAL(
        shape.size() > 1,
        "Full operation error: Shape size must be greater than 1, but got shape size = {}",
        shape.size());

    for (size_t i = 0; i < shape.size(); i++) {
        TT_FATAL(
            shape[i] > 0,
            "Full operation error: Invalid shape at index {}. Each dimension of the shape must be greater than 0, but"
            "got shape[{}] = {}",
            i,
            i,
            shape[i]);
    }
}

void FullDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

FullDeviceOperation::spec_return_value_t FullDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    return TensorSpec(
        Shape(operation_attributes.shape),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
};

FullDeviceOperation::tensor_return_value_t FullDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, operation_attributes.mesh_device);
}

}  // namespace ttnn::operations::full

namespace ttnn::prim {
ttnn::operations::full::FullDeviceOperation::tensor_return_value_t full(
    ttnn::SmallVector<uint32_t> shape,
    std::variant<float, int> fill_value,
    ttnn::MeshDevice* mesh_device,
    const DataType& dtype,
    const Layout& layout,
    const MemoryConfig& memory_config) {
    using OperationType = ttnn::operations::full::FullDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        std::move(shape),
        fill_value,
        mesh_device,
        dtype,
        layout,
        memory_config,
    };
    auto tensor_args = OperationType::tensor_args_t{};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
