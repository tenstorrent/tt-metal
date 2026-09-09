// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

ReshapeCodegenDeviceOperation::program_factory_t ReshapeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        return ReshapeCodegenRmProgramFactory{};
    }
    return ReshapeCodegenTileProgramFactory{};
}

void ReshapeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to ReshapeCodegen need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to ReshapeCodegen need to be allocated in buffers on device!");
    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::TILE || input.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Only tile and row major ReshapeCodegen supported!");
    TT_FATAL(input.padded_shape().rank() >= 1, "ReshapeCodegen requires rank >= 1 input");
    TT_FATAL(tensor_args.output_padded_shape.rank() >= 1, "ReshapeCodegen requires rank >= 1 output");

    TT_FATAL(
        ttnn::operations::data_movement::reshape_codegen::supported_by_codegen(
            input,
            tensor_args.output_logical_shape,
            tensor_args.output_padded_shape,
            operation_attributes.output_mem_config),
        "Input is not supported by ReshapeCodegen");

    if (tensor_args.optional_output_tensor.has_value()) {
        const auto& out = tensor_args.optional_output_tensor.value();
        TT_FATAL(
            out.logical_shape() == tensor_args.output_logical_shape, "ReshapeCodegen optional output shape mismatch");
        TT_FATAL(out.dtype() == input.dtype(), "ReshapeCodegen optional output dtype mismatch");
        TT_FATAL(out.layout() == input.layout(), "ReshapeCodegen optional output layout mismatch");
        TT_FATAL(out.device() == input.device(), "ReshapeCodegen optional output must be on the same device");
        TT_FATAL(out.buffer() != nullptr, "ReshapeCodegen optional output must be allocated in a buffer on device!");
    }
}

ReshapeCodegenDeviceOperation::spec_return_value_t ReshapeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        tensor_args.output_logical_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input.dtype(),
            input.tensor_spec().page_config(),
            operation_attributes.output_mem_config,
            tensor_args.output_logical_shape,
            tensor_args.output_padded_shape));
}

ReshapeCodegenDeviceOperation::tensor_return_value_t ReshapeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> ReshapeCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

ReshapeCodegenDeviceOperation::tensor_return_value_t reshape_codegen(
    const Tensor& input,
    const ttnn::Shape& output_logical_shape,
    const ttnn::Shape& output_padded_shape,
    const ReshapeCodegenParams& params,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = ReshapeCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input = input,
            .output_logical_shape = output_logical_shape,
            .output_padded_shape = output_padded_shape,
            .optional_output_tensor = std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
