// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
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
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    const auto& input_tensor = tensor_args.input;
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    // Same scheme as native UntilizeDeviceOperation::compute_output_specs: the output keeps
    // the input's PADDED shape (so a non-tile-aligned logical shape still gets a physically
    // tile-row/column-aligned buffer the program factory can write in full tile-rows), with
    // only the logical_shape metadata cropping it back down on read.
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
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
