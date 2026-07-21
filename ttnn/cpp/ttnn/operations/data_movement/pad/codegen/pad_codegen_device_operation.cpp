// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "pad_codegen_supported.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

void PadCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        supported_by_codegen(operation_attributes, tensor_args),
        "ttnn.pad (codegen): input is not supported by the codegen path");
}

PadCodegenDeviceOperation::spec_return_value_t PadCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input;
    const ttnn::Shape output_logical_shape(
        {operation_attributes.N_out,
         operation_attributes.C_out,
         operation_attributes.H_out,
         operation_attributes.W_out});
    // RM: physical storage is exactly the logical shape (no tile rounding). TILE: the physical
    // carrier is the tile-page ceiling of H_out/W_out (PadCodegenParams stores H_out/W_out in
    // element units for both layouts -- see pad_codegen_device_operation_types.hpp), matching
    // ordinary TILE_LAYOUT tensors whose logical shape need not be tile-aligned.
    ttnn::Shape output_padded_shape = output_logical_shape;
    if (input_tensor.layout() == Layout::TILE) {
        output_padded_shape = ttnn::Shape(
            {operation_attributes.N_out,
             operation_attributes.C_out,
             tt::div_up(operation_attributes.H_out, TILE_HEIGHT) * TILE_HEIGHT,
             tt::div_up(operation_attributes.W_out, TILE_WIDTH) * TILE_WIDTH});
    }
    return TensorSpec(
        output_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config,
            output_logical_shape,
            output_padded_shape));
}

PadCodegenDeviceOperation::tensor_return_value_t PadCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

Tensor pad_codegen(
    const Tensor& input,
    const PadCodegenDeviceOperation::operation_attributes_t& operation_attributes,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = PadCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        operation_attributes, OperationType::tensor_args_t{input, preallocated_output});
}

}  // namespace ttnn::prim
