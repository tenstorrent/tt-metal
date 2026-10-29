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
    TT_FATAL(input.is_allocated(), "Operands need to be allocated in buffers on device!");
}

}  // namespace

ReshapeCodegenDeviceOperation::program_factory_t ReshapeCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    namespace rcg = ttnn::operations::data_movement::reshape_codegen;
    // The live choice, not the planned route: a tile-rows call too big for live L1 runs the segment map.
    const auto chosen = rcg::live_tile_choice(
        tensor_args.input,
        operation_attributes.logical_output_shape,
        rcg::reshape_core_grid(tensor_args.input, operation_attributes.sub_core_grids).num_cores(),
        operation_attributes.plan);
    if (chosen.has_value() && std::holds_alternative<rcg::TileMappedChoice>(*chosen)) {
        return ReshapeCodegenTileMappedProgramFactory{};
    }
    if (operation_attributes.plan.tile_route == rcg::TileRoute::TileRows) {
        return ReshapeCodegenTileRowsProgramFactory{};
    }
    return ReshapeCodegenTileMappedProgramFactory{};
}

void ReshapeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    namespace rcg = ttnn::operations::data_movement::reshape_codegen;
    const Tensor& input = tensor_args.input;
    require_device_input(input);
    const auto& plan = operation_attributes.plan;
    TT_FATAL(
        rcg::supported_by_codegen(
            input,
            operation_attributes.logical_output_shape,
            operation_attributes.output_mem_config,
            operation_attributes.sub_core_grids,
            plan),
        "ReshapeCodegen: {}",
        rcg::unsupported_reason(
            input,
            operation_attributes.logical_output_shape,
            operation_attributes.output_mem_config,
            operation_attributes.sub_core_grids,
            &plan));
    if (plan.tile_route != rcg::TileRoute::TileRows) {
        return;
    }
    const auto& layout = plan.tile_rows.layout;
    // A view keeps its source's buffer, so the buffer may hold more pages than the plan reads; interleaved
    // page i sits at the same address either way, so only a short buffer is an error.
    TT_FATAL(
        input.buffer()->num_pages() >= layout.num_input_pages,
        "Reshape input has {} tiles, the tile-row plan needs at least {}",
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

// The hashed rung or depth steps along fixed ladders (tile-row rungs, segment-map input depths), so the key changes
// only when live L1 crosses a rung or tile rows falls back to the segment map. A call no plan fits hashes rung 0,
// which never has an entry: it misses and the program factory raises.
ttsl::hash::hash_t ReshapeCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& plan = operation_attributes.plan;
    namespace rcg = ttnn::operations::data_movement::reshape_codegen;
    const auto chosen = rcg::live_tile_choice(
        tensor_args.input,
        operation_attributes.logical_output_shape,
        rcg::reshape_core_grid(tensor_args.input, operation_attributes.sub_core_grids).num_cores(),
        plan);
    // Tile rows: rung and staging depth. Segment map: slicing and input CB depth.
    size_t program = 0;
    uint32_t first = 0;
    uint32_t second = 0;
    if (chosen.has_value()) {
        program = chosen->index();
        if (const auto* rows = std::get_if<rcg::TileRowsChoice>(&*chosen)) {
            first = rows->max_in;
            second = rows->staged_batches;
        } else {
            const auto& mapped = std::get<rcg::TileMappedChoice>(*chosen);
            first = mapped.layout.sliced ? 1 : 0;
            second = mapped.input_depth;
        }
    }
    return ttsl::hash::hash_objects_with_default_seed(
        operation_attributes, tensor_args, plan.tile_route, program, first, second);
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
