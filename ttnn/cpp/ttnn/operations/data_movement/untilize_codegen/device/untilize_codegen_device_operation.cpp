// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/untilize_codegen/device/untilize_codegen_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

UntilizeCodegenDeviceOperation::program_factory_t UntilizeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return UntilizeCodegenProgramFactory{};
}

void UntilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "untilize_codegen: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "untilize_codegen: input must be allocated");
    TT_FATAL(input.layout() == tt::tt_metal::Layout::TILE, "untilize_codegen: TILE layout only");
    TT_FATAL(input.dtype() == tt::tt_metal::DataType::BFLOAT16, "untilize_codegen: bfloat16 only");
    TT_FATAL(!input.is_sharded(), "untilize_codegen: interleaved input only");
}

UntilizeCodegenDeviceOperation::spec_return_value_t UntilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Matches ttnn.untilize: same shape, same dtype, layout flips TILE -> ROW_MAJOR.
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            input.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            operation_attributes.m_output_mem_config));
}

UntilizeCodegenDeviceOperation::tensor_return_value_t UntilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

UntilizeCodegenDeviceOperation::tensor_return_value_t untilize_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = UntilizeCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.m_output_mem_config = output_mem_config},
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
