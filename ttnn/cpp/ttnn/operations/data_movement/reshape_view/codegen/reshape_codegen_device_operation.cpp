// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_device_operation.hpp"

#include <variant>

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
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input.layout() == Layout::TILE) {
        return ReshapeCodegenTileRowsProgramFactory{};
    }
    return ReshapeCodegenProgramFactory{};
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
    const auto& plan = operation_attributes.plan;
    if (input.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            plan.rm.has_value() || plan.rm_arbitrary.has_value(),
            "ReshapeCodegen was launched without a row-major geometry");
        return;
    }
    const auto& layout = plan.tile_rows.layout;
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

// With the full plan fitting, the hashed rung or depth is the same constant on every call, so calls share programs
// exactly as the attributes and tensor args alone would key them. A call no plan fits hashes rung 0, which never has
// an entry: it misses and the program factory raises.
ttsl::hash::hash_t ReshapeCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& plan = operation_attributes.plan;
    if (tensor_args.input.layout() == Layout::TILE) {
        const auto chosen = ttnn::operations::data_movement::reshape_codegen::live_tile_rows_choice(
            tensor_args.input, operation_attributes.logical_output_shape, plan.tile_rows);
        const uint32_t max_in = chosen ? chosen->max_in : 0;
        const uint32_t staged_batches = chosen ? chosen->staged_batches : 0;
        return ttsl::hash::hash_objects_with_default_seed(operation_attributes, tensor_args, max_in, staged_batches);
    }
    const auto chosen = ttnn::operations::data_movement::reshape_codegen::live_rm_choice(
        tensor_args.input,
        operation_attributes.logical_output_shape,
        operation_attributes.output_mem_config.buffer_type(),
        ttnn::operations::data_movement::reshape_codegen::reshape_core_grid(
            tensor_args.input, operation_attributes.sub_core_grids)
            .num_cores(),
        plan);
    // Depth 0 is never built, so a call no program fits misses like a tile-rows call at rung 0.
    uint32_t cb_depth = 0;
    uint32_t push_sticks = 0;
    bool slab = false;
    if (chosen.has_value()) {
        if (const auto* stick = std::get_if<ttnn::operations::data_movement::reshape_codegen::RmPlan>(&*chosen)) {
            cb_depth = stick->cb_depth;
            push_sticks = stick->push_sticks;
        } else {
            cb_depth = std::get<ttnn::operations::data_movement::reshape_codegen::RmArbitraryPlan>(*chosen).batch;
            slab = true;
        }
    }
    return ttsl::hash::hash_objects_with_default_seed(operation_attributes, tensor_args, cb_depth, push_sticks, slab);
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
