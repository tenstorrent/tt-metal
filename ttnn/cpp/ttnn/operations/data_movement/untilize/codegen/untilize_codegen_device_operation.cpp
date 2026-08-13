// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/untilize/codegen/untilize_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/untilize/codegen/untilize_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

UntilizeCodegenDeviceOperation::program_factory_t UntilizeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return UntilizeCodegenProgramFactory{};
}

void UntilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const Tensor& input = tensor_args.input;

    // Native's own structural TT_FATALs -- supported_by_codegen() only asks
    // about layout/dtype/memory-config, all of which answer fine for a
    // host-side or deallocated tensor, so these must be checked here too.
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input.layout() == Layout::TILE, "Can only untilize tile major data");

    uint32_t tensor_width = input.padded_shape()[-1];
    uint32_t tensor_height = input.physical_volume() / tensor_width;
    TT_FATAL(tensor_width % TILE_WIDTH == 0, "Width must be evenly divisible into tiles");
    TT_FATAL(tensor_height % TILE_HEIGHT == 0, "Height must be evenly divisible into tiles");

    TT_FATAL(
        ttnn::operations::data_movement::untilize_codegen::supported_by_codegen(input),
        "Input is not supported by UntilizeCodegen");
}

UntilizeCodegenDeviceOperation::spec_return_value_t UntilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            output_dtype,
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

UntilizeCodegenDeviceOperation::tensor_return_value_t UntilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor>
UntilizeCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

UntilizeCodegenDeviceOperation::tensor_return_value_t untilize_codegen(
    const Tensor& input, const UntilizeCodegenParams& params) {
    using OperationType = UntilizeCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
