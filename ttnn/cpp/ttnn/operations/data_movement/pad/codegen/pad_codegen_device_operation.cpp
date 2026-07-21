// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_device_operation.hpp"

#include "pad_codegen_supported.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

void PadCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        supported_by_codegen(operation_attributes, tensor_args),
        "ttnn.pad (codegen): input is not supported by the codegen path");
}

PadCodegenDeviceOperation::spec_return_value_t PadCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    TT_THROW("pad_codegen: compute_output_specs not yet implemented");
}

PadCodegenDeviceOperation::tensor_return_value_t PadCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    TT_THROW("pad_codegen: create_output_tensors not yet implemented");
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
