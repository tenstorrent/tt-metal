// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_device_operation.hpp"

#include <tt-metalium/constants.hpp>

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

    const auto& logical_shape = input_tensor.logical_shape();
    const bool tile_aligned =
        logical_shape[-2] % tt::constants::TILE_HEIGHT == 0 && logical_shape[-1] % tt::constants::TILE_WIDTH == 0;
    if (!tile_aligned) {
        // Mirrors native UntilizeWithUnpaddingDeviceOperation::compute_output_specs: the writer
        // (build_with_unpadding) strips physical tile padding down to the logical shape, so the
        // output tensor is genuinely compact -- padded_shape == logical_shape, not the input's
        // tile-grid-rounded padded_shape.
        return TensorSpec(
            logical_shape,
            TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), operation_attributes.output_mem_config));
    }

    // Tile-aligned: same scheme as native UntilizeDeviceOperation::compute_output_specs. The
    // output keeps the input's PADDED shape (so the program factory can write in full physical
    // tile-rows), with only the logical_shape metadata cropping it back down on read.
    return TensorSpec(
        logical_shape,
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            logical_shape,
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
