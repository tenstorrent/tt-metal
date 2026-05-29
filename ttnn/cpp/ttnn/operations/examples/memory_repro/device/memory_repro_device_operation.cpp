// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_repro_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::examples {

MemoryReproDeviceOperation::program_factory_t MemoryReproDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void MemoryReproDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

MemoryReproDeviceOperation::spec_return_value_t MemoryReproDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), MemoryConfig{}));
}

MemoryReproDeviceOperation::tensor_return_value_t MemoryReproDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::examples

namespace ttnn::prim {
ttnn::operations::examples::MemoryReproDeviceOperation::tensor_return_value_t memory_repro(const Tensor& input_tensor) {
    using OperationType = ttnn::operations::examples::MemoryReproDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{}, OperationType::tensor_args_t{input_tensor});
}
}  // namespace ttnn::prim
