// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

ReshapeCodegenDeviceOperation::program_factory_t ReshapeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ReshapeCodegenProgramFactory{};
}

void ReshapeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        ttnn::operations::data_movement::reshape_codegen::supported_by_codegen(
            input, operation_attributes.output_shape[-1], operation_attributes.output_mem_config),
        "Input is not supported by the generated reshape implementation");
}

ReshapeCodegenDeviceOperation::spec_return_value_t ReshapeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        operation_attributes.output_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout()), operation_attributes.output_mem_config));
}

ReshapeCodegenDeviceOperation::tensor_return_value_t ReshapeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ReshapeCodegenDeviceOperation::tensor_return_value_t reshape_codegen(
    const Tensor& input, const ReshapeCodegenParams& params) {
    using OperationType = ReshapeCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
