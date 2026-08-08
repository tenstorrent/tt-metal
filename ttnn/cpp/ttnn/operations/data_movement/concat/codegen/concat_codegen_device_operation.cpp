// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

ConcatCodegenDeviceOperation::program_factory_t ConcatCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ConcatCodegenProgramFactory{};
}

void ConcatCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "ConcatCodegen needs 1 or more input tensors!");
    const auto& first_input = input_tensors.at(0);
    auto shape_first = first_input.logical_shape();
    TT_FATAL(
        operation_attributes.dim < shape_first.rank(), "ConcatCodegen dim specified is larger than input tensor rank.");
    shape_first[operation_attributes.dim] = 0;
    for (const auto& input : input_tensors) {
        TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to concat need to be on device!");
        TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
        TT_FATAL(input.device() == first_input.device(), "Operands to concat need to be on the same device!");
        TT_FATAL(input.layout() == first_input.layout(), "All Tensors should have same layouts.");
        TT_FATAL(input.dtype() == first_input.dtype(), "All Tensors should have same dtypes.");
        TT_FATAL(input.logical_shape().rank() == shape_first.rank(), "ConcatCodegen input tensor ranks must be equal");
        auto curr_shape = input.logical_shape();
        curr_shape[operation_attributes.dim] = 0;
        TT_FATAL(curr_shape == shape_first, "ConcatCodegen tensors differ in shape across non-concat dimensions.");
    }
    TT_FATAL(
        ttnn::operations::data_movement::concat_codegen::supported_by_codegen(
            input_tensors, operation_attributes.dim, operation_attributes.output_mem_config),
        "Input is not supported by ConcatCodegen");
}

ConcatCodegenDeviceOperation::spec_return_value_t ConcatCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensors;
    const Tensor& ref_input = input_tensors.at(0);
    ttnn::Shape output_shape = ref_input.logical_shape();
    output_shape[operation_attributes.dim] = 0;
    for (const auto& input : input_tensors) {
        output_shape[operation_attributes.dim] += input.logical_shape()[operation_attributes.dim];
    }
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            ref_input.dtype(), tt::tt_metal::PageConfig(ref_input.layout()), operation_attributes.output_mem_config));
}

ConcatCodegenDeviceOperation::tensor_return_value_t ConcatCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensors.at(0).device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> ConcatCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensors.at(0);
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

ConcatCodegenDeviceOperation::tensor_return_value_t concat_codegen(
    const std::vector<Tensor>& input_tensors, const ConcatCodegenParams& params) {
    using OperationType = ConcatCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params, OperationType::tensor_args_t{.input_tensors = input_tensors});
}

}  // namespace ttnn::prim
