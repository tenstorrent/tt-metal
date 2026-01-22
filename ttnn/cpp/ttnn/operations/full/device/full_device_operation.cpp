// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full {
void FullOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(
        operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Full operation error: Not currently supporting sharding");

    const auto shape = operation_attributes.shape;

    TT_FATAL(
        shape.size() > 1,
        "Full operation error: Shape size must be greater than 1, but got shape size = {}",
        shape.size());

    for (int i = 0; i < shape.size(); i++) {
        TT_FATAL(
            shape[i] > 0,
            "Full operation error: Invalid shape at index {}. Each dimension of the shape must be greater than 0, but"
            "got shape[{}] = {}",
            i,
            i,
            shape[i]);
    }
}

FullOperation::program_factory_t FullOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ProgramFactory{};
}

void FullOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void FullOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

FullOperation::spec_return_value_t FullOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    return TensorSpec(
        Shape(operation_attributes.shape),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
};

FullOperation::tensor_return_value_t FullOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, operation_attributes.mesh_device);
}

}  // namespace ttnn::operations::full

namespace ttnn::prim {
ttnn::operations::full::FullOperation::tensor_return_value_t full(
    ttnn::SmallVector<uint32_t> shape,
    std::variant<float, int> fill_value,
    ttnn::MeshDevice* mesh_device,
    const DataType& dtype,
    const Layout& layout,
    const MemoryConfig& memory_config) {
    using OperationType = ttnn::operations::full::FullOperation;
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
