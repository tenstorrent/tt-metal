// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

namespace {

// The prim body and a cache hit reach the device before any other check.
void require_device_input(const Tensor& input) {
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
}

}  // namespace

ReshapeCodegenDeviceOperation::program_factory_t ReshapeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ReshapeCodegenTileRowsProgramFactory{};
}

void ReshapeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    namespace rcg = ttnn::operations::data_movement::reshape_codegen;
    const Tensor& input = tensor_args.input;
    require_device_input(input);
    const char* reason = rcg::unsupported_reason(
        input,
        operation_attributes.logical_output_shape,
        operation_attributes.output_mem_config,
        operation_attributes.sub_core_grids);
    TT_FATAL(reason == nullptr, "ReshapeCodegen: {}", reason);
    const auto& layout = operation_attributes.plan.tile_rows.layout;
    TT_FATAL(
        input.buffer()->num_pages() == layout.num_input_pages,
        "Reshape input has {} tiles, the tile-row plan expects {}",
        input.buffer()->num_pages(),
        layout.num_input_pages);
}

void ReshapeCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    require_device_input(tensor_args.input);
}

ReshapeCodegenDeviceOperation::spec_return_value_t ReshapeCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ttnn::operations::data_movement::reshape_codegen::output_spec(
        tensor_args.input, operation_attributes.logical_output_shape, operation_attributes.output_mem_config);
}

ReshapeCodegenDeviceOperation::tensor_return_value_t ReshapeCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor reshape_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    ttnn::operations::data_movement::reshape_codegen::LaunchPlan plan) {
    using OperationType = ReshapeCodegenDeviceOperation;
    require_device_input(input);
    return ttnn::device_operation::launch<OperationType>(
        ReshapeCodegenParams{
            .logical_output_shape = logical_output_shape,
            .output_mem_config = output_mem_config,
            .sub_core_grids = sub_core_grids,
            .plan = std::move(plan)},
        OperationType::tensor_args_t{.input = input});
}

}  // namespace ttnn::prim
