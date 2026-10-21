// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

ConcatCodegenDeviceOperation::program_factory_t ConcatCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ConcatCodegenProgramFactory{};
}

void ConcatCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(!tensor_args.input_tensors.empty(), "need 1 or more tensors");
    TT_FATAL(
        ttnn::operations::data_movement::concat_codegen::supported_by_codegen(
            tensor_args.input_tensors, operation_attributes),
        "Input is not supported by ConcatCodegen");
}

ConcatCodegenDeviceOperation::spec_return_value_t ConcatCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& first = tensor_args.input_tensors.at(0);
    ttnn::Shape shape_out = first.logical_shape();
    shape_out[operation_attributes.dim] = 0;
    for (const Tensor& input : tensor_args.input_tensors) {
        shape_out[operation_attributes.dim] += input.logical_shape()[operation_attributes.dim];
    }
    return tt::tt_metal::TensorSpec(
        shape_out,
        tt::tt_metal::TensorLayout(
            first.dtype(), tt::tt_metal::PageConfig(first.layout()), operation_attributes.output_mem_config));
}

ConcatCodegenDeviceOperation::tensor_return_value_t ConcatCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensors.at(0).device());
}

ConcatCodegenDeviceOperation::tensor_return_value_t concat_codegen(
    const std::vector<Tensor>& input_tensors, const ConcatCodegenParams& params) {
    using OperationType = ConcatCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input_tensors = input_tensors});
}

}  // namespace ttnn::prim
