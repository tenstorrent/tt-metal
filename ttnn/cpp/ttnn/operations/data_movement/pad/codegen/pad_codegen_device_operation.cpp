// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

PadCodegenDeviceOperation::program_factory_t PadCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return PadCodegenProgramFactory{};
}

void PadCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to pad need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        ttnn::operations::data_movement::pad_codegen::supported_by_codegen(
            input, operation_attributes.output_mem_config),
        "Input is not supported by PadCodegen");
}

PadCodegenDeviceOperation::spec_return_value_t PadCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        operation_attributes.output_logical_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout()), operation_attributes.output_mem_config));
}

PadCodegenDeviceOperation::tensor_return_value_t PadCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> PadCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

PadCodegenDeviceOperation::tensor_return_value_t pad_codegen(
    const Tensor& input, const PadCodegenParams& params, std::optional<Tensor> optional_output_tensor) {
    using OperationType = PadCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{.input = input, .optional_output_tensor = std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
