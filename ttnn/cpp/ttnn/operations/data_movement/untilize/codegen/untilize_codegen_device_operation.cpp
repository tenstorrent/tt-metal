// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "untilize_codegen_supported.hpp"

namespace ttnn::prim {

void UntilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        ttnn::operations::data_movement::untilize_codegen::supported_by_codegen(
            tensor_args.input, operation_attributes.output_mem_config),
        "UntilizeCodegenDeviceOperation invoked for a case not supported by the codegen implementation");
}

UntilizeCodegenDeviceOperation::spec_return_value_t UntilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.input.tensor_spec();
}

UntilizeCodegenDeviceOperation::tensor_return_value_t UntilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor untilize_codegen(const Tensor& input, tt::tt_metal::MemoryConfig output_mem_config) {
    return ttnn::device_operation::launch<UntilizeCodegenDeviceOperation>(
        UntilizeCodegenOperationAttributes{.output_mem_config = std::move(output_mem_config)},
        UntilizeCodegenTensorArgs{.input = input});
}

}  // namespace ttnn::prim
