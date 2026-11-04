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

}  // namespace

void TilizeCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    codegen::require_allocated_device_input(input);
    TT_FATAL(
        codegen::supported_by_codegen(input, operation_attributes.output_mem_config, operation_attributes.output_dtype),
        "TilizeCodegenDeviceOperation invoked for a case not supported by the codegen implementation");

    // The factory and kernels trust the plan; these are the properties they rely on.
    const auto& plan = operation_attributes.plan;
    const auto geometry = codegen::tilize_geometry(input, operation_attributes.output_dtype);
    TT_FATAL(
        plan.chunk_wt > 0 && plan.chunk_wt * plan.num_col_chunks == geometry.block_wt,
        "Tilize codegen plan: {} chunks of {} tiles do not cover the {}-tile column block",
        plan.num_col_chunks,
        plan.chunk_wt,
        geometry.block_wt);
    // Readers, compute and the typecast stage move whole chunks as contiguous page runs from the FIFO pointer, so a
    // chunk must never straddle the end of a CB.
    TT_FATAL(
        plan.cb_in_depth >= plan.chunk_wt && plan.cb_in_depth % plan.chunk_wt == 0 &&
            plan.cb_out_depth >= plan.chunk_wt && plan.cb_out_depth % plan.chunk_wt == 0,
        "Tilize codegen plan: CB depths (in {}, out {} tiles) must be non-zero multiples of the {}-tile chunk",
        plan.cb_in_depth,
        plan.cb_out_depth,
        plan.chunk_wt);
    if (plan.write_batch > 1) {
        // The batched writer addresses each batch from the FIFO read pointer without wrapping, which holds only
        // when the core writes a single block that fits the output CB.
        const auto grid = input.device()->compute_with_storage_grid_size();
        const bool one_row_per_core = geometry.route == codegen::TilizeRoute::Column2D ||
                                      geometry.tile_rows <= static_cast<uint32_t>(grid.x * grid.y);
        TT_FATAL(
            one_row_per_core && plan.cb_out_depth >= geometry.block_wt,
            "Tilize codegen plan: batched writes need one tile row per core and an output CB of at least {} tiles, "
            "got {}",
            geometry.block_wt,
            plan.cb_out_depth);
    }
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
    const codegen::TilizeCodegenPlan& plan) {
    codegen::require_allocated_device_input(input);
    return ttnn::device_operation::launch<TilizeCodegenDeviceOperation>(
        TilizeCodegenOperationAttributes{
            .output_mem_config = std::move(output_mem_config), .output_dtype = output_dtype, .plan = plan},
        TilizeCodegenTensorArgs{.input = input});
}

}  // namespace ttnn::prim
