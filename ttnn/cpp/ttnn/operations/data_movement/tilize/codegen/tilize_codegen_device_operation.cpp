// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "tilize_codegen_supported.hpp"

namespace ttnn::prim {

namespace {

namespace codegen = ttnn::operations::data_movement::tilize_codegen;

// Answering the support question for a host tensor would dereference a null device, so the structural
// preconditions are asserted first.
void require_allocated_device_input(const Tensor& input) {
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(input.is_allocated(), "Operands to tilize need to be allocated in buffers on device!");
}

}  // namespace

void TilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    require_allocated_device_input(input);
    TT_FATAL(
        codegen::supported_by_codegen(input, operation_attributes.output_mem_config, operation_attributes.output_dtype),
        "TilizeCodegenDeviceOperation invoked for a case not supported by the codegen implementation");
    const auto unsupported_control = codegen::unsupported_execution_control(
        operation_attributes.use_multicore,
        operation_attributes.use_low_perf,
        input.tensor_spec().tile(),
        operation_attributes.sub_core_grids);
    TT_FATAL(
        !unsupported_control.has_value(),
        "The codegen tilize runs on the full compute grid with the default tile and cannot honour {}",
        unsupported_control.value_or(""));
}

TilizeCodegenDeviceOperation::spec_return_value_t TilizeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(
            operation_attributes.output_dtype,
            PageConfig(Layout::TILE, input.tensor_spec().tile()),
            operation_attributes.output_mem_config));
}

TilizeCodegenDeviceOperation::tensor_return_value_t TilizeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor tilize_codegen(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    tt::tt_metal::DataType output_dtype,
    bool use_multicore,
    bool use_low_perf,
    bool preserve_logical_shape,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const codegen::TilizeCodegenPlan& plan) {
    require_allocated_device_input(input);
    return ttnn::device_operation::launch<TilizeCodegenDeviceOperation>(
        TilizeCodegenOperationAttributes{
            .output_mem_config = std::move(output_mem_config),
            .output_dtype = output_dtype,
            .use_multicore = use_multicore,
            .use_low_perf = use_low_perf,
            .preserve_logical_shape = preserve_logical_shape,
            .sub_core_grids = sub_core_grids,
            .plan = plan},
        TilizeCodegenTensorArgs{.input = input});
}

}  // namespace ttnn::prim
