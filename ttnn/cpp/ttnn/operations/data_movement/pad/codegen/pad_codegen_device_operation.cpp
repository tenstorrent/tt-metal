// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>
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
    using namespace tt::constants;
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operand to pad needs to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::TILE || input.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "PadCodegen only supports TILE or ROW_MAJOR input layout");
    TT_FATAL(input.logical_shape().rank() == 4, "PadCodegen requires a 4D input");

    const auto& in_shape = input.logical_shape();
    // Structural bounds checks matching native's own (input + front <=
    // output, in each dim), stated in each layout's own units.
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        TT_FATAL(
            in_shape[0] + operation_attributes.front_n <= operation_attributes.N_out, "Output size cannot fit input with offset");
        TT_FATAL(
            in_shape[1] + operation_attributes.front_c <= operation_attributes.C_out, "Output size cannot fit input with offset");
        TT_FATAL(
            in_shape[2] + operation_attributes.front_h <= operation_attributes.H_out, "Output size cannot fit input with offset");
        TT_FATAL(
            in_shape[3] + operation_attributes.front_w <= operation_attributes.W_out, "Output size cannot fit input with offset");
    } else {
        const uint32_t Ht_in = input.padded_shape()[2] / TILE_HEIGHT;
        const uint32_t Wt_in = input.padded_shape()[3] / TILE_WIDTH;
        TT_FATAL(
            in_shape[0] + operation_attributes.front_n <= operation_attributes.N_out, "Output size cannot fit input with offset");
        TT_FATAL(
            in_shape[1] + operation_attributes.front_c <= operation_attributes.C_out, "Output size cannot fit input with offset");
        TT_FATAL(Ht_in + operation_attributes.front_h <= operation_attributes.H_out, "Output size cannot fit input with offset");
        TT_FATAL(Wt_in + operation_attributes.front_w <= operation_attributes.W_out, "Output size cannot fit input with offset");
        TT_FATAL(
            operation_attributes.front_n == 0 && operation_attributes.front_c == 0 &&
                operation_attributes.front_h == 0 && operation_attributes.front_w == 0,
            "PadCodegen TILE branch only supports padding at end of dims");
    }

    const std::array<uint32_t, 4> front = {
        operation_attributes.front_n, operation_attributes.front_c, operation_attributes.front_h,
        operation_attributes.front_w};
    TT_FATAL(
        ttnn::operations::data_movement::pad_codegen::supported_by_codegen(
            input, operation_attributes.output_padded_shape, front, operation_attributes.output_mem_config),
        "Input is not supported by PadCodegen");
}

PadCodegenDeviceOperation::spec_return_value_t PadCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        operation_attributes.output_logical_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input.dtype(),
            tt::tt_metal::PageConfig(input.layout()),
            operation_attributes.output_mem_config,
            operation_attributes.output_logical_shape,
            operation_attributes.output_padded_shape));
}

PadCodegenDeviceOperation::tensor_return_value_t PadCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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

PadCodegenDeviceOperation::tensor_return_value_t pad_codegen(const Tensor& input, const PadCodegenParams& params) {
    using OperationType = PadCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(params, OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
