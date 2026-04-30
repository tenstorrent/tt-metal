// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {
void FastReduceNCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& preallocated_output = tensor_args.preallocated_output;

    // FLOAT32 is allowed so multi-stage Sum chains can carry FP32 between stages.
    operations::check_tensor(
        input, "FastReduceNC", "input", {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::FLOAT32});
    if (preallocated_output.has_value()) {
        operations::check_tensor(
            preallocated_output.value(),
            "FastReduceNC",
            "output",
            {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::FLOAT32});
        TT_FATAL(
            preallocated_output.value().logical_shape().rank() == input.logical_shape().rank(),
            "FastReduceNC preallocated output rank {} must match input rank {}",
            preallocated_output.value().logical_shape().rank(),
            input.logical_shape().rank());
        const auto& preallocated_output_tensor = preallocated_output.value();
        const uint32_t pre_out_tile_height = preallocated_output_tensor.tensor_spec().tile().get_height();
        const uint32_t pre_out_tile_width = preallocated_output_tensor.tensor_spec().tile().get_width();
        const auto& pre_out_padded_shape = preallocated_output_tensor.padded_shape();
        TT_FATAL(
            pre_out_padded_shape.rank() >= 2,
            "FastReduceNC preallocated output padded_shape rank {} must be at least 2",
            pre_out_padded_shape.rank());
        TT_FATAL(
            pre_out_padded_shape[-2] > 0 && pre_out_padded_shape[-1] > 0,
            "FastReduceNC preallocated output padded last-2 dims must be positive");
        TT_FATAL(
            pre_out_padded_shape[-2] % pre_out_tile_height == 0,
            "FastReduceNC preallocated output padded height {} must be tile-height-aligned ({})",
            pre_out_padded_shape[-2],
            pre_out_tile_height);
        TT_FATAL(
            pre_out_padded_shape[-1] % pre_out_tile_width == 0,
            "FastReduceNC preallocated output padded width {} must be tile-width-aligned ({})",
            pre_out_padded_shape[-1],
            pre_out_tile_width);
        TT_FATAL(
            preallocated_output_tensor.physical_volume() % (pre_out_tile_height * pre_out_tile_width) == 0,
            "FastReduceNC preallocated output physical volume must be a multiple of tile element count {} (got {})",
            pre_out_tile_height * pre_out_tile_width,
            preallocated_output_tensor.physical_volume());
    }

    // validate input dim
    const auto input_rank = input.logical_shape().rank();
    TT_FATAL(
        (args.dim >= 0 && args.dim <= tt::tt_metal::MAX_NUM_DIMENSIONS - 2),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS - 2);
    TT_FATAL((args.dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
    TT_FATAL(
        input_rank <= tt::tt_metal::MAX_NUM_DIMENSIONS,
        "FastReduceNC input rank {} exceeds maximum {}",
        input_rank,
        tt::tt_metal::MAX_NUM_DIMENSIONS);

    {
        const uint32_t in_tile_height = input.tensor_spec().tile().get_height();
        const uint32_t in_tile_width = input.tensor_spec().tile().get_width();
        const auto& in_padded_shape = input.padded_shape();
        TT_FATAL(
            in_padded_shape.rank() >= 2,
            "FastReduceNC input padded_shape rank {} must be at least 2",
            in_padded_shape.rank());
        TT_FATAL(
            in_padded_shape[-2] > 0 && in_padded_shape[-1] > 0,
            "FastReduceNC input padded last-2 dims must be positive");
        TT_FATAL(
            in_padded_shape[-2] % in_tile_height == 0,
            "FastReduceNC input padded height {} must be tile-height-aligned ({})",
            in_padded_shape[-2],
            in_tile_height);
        TT_FATAL(
            in_padded_shape[-1] % in_tile_width == 0,
            "FastReduceNC input padded width {} must be tile-width-aligned ({})",
            in_padded_shape[-1],
            in_tile_width);
        TT_FATAL(
            input.physical_volume() % (in_tile_height * in_tile_width) == 0,
            "FastReduceNC input physical volume must be a multiple of tile element count {} (got {})",
            in_tile_height * in_tile_width,
            input.physical_volume());
    }

    const auto fr_nc_device_grid = tensor_args.input.device()->compute_with_storage_grid_size();
    TT_FATAL(
        fr_nc_device_grid.x > 0 && fr_nc_device_grid.y > 0,
        "FastReduceNC requires non-empty device compute grid, got ({}, {})",
        fr_nc_device_grid.x,
        fr_nc_device_grid.y);
    const tt::tt_metal::CoreRangeSet fr_nc_full_device_grid =
        tt::tt_metal::num_cores_to_corerangeset(fr_nc_device_grid.x * fr_nc_device_grid.y, fr_nc_device_grid, true);
    const auto& fr_nc_in_memory_config = tensor_args.input.memory_config();
    if (fr_nc_in_memory_config.shard_spec().has_value()) {
        TT_FATAL(
            fr_nc_full_device_grid.contains(fr_nc_in_memory_config.shard_spec().value().grid),
            "FastReduceNC input shard grid {} must be contained in device compute grid {}",
            fr_nc_in_memory_config.shard_spec().value().grid,
            fr_nc_full_device_grid);
        const auto& in_shard_spec = fr_nc_in_memory_config.shard_spec().value();
        const uint32_t in_shard_tile_height = input.tensor_spec().tile().get_height();
        const uint32_t in_shard_tile_width = input.tensor_spec().tile().get_width();
        TT_FATAL(
            in_shard_spec.shape[0] > 0 && in_shard_spec.shape[1] > 0,
            "FastReduceNC input shard_shape must be positive, got [{}, {}]",
            in_shard_spec.shape[0],
            in_shard_spec.shape[1]);
        TT_FATAL(
            in_shard_spec.shape[0] % in_shard_tile_height == 0,
            "FastReduceNC input shard_shape[0]={} must be tile-height-aligned ({})",
            in_shard_spec.shape[0],
            in_shard_tile_height);
        TT_FATAL(
            in_shard_spec.shape[1] % in_shard_tile_width == 0,
            "FastReduceNC input shard_shape[1]={} must be tile-width-aligned ({})",
            in_shard_spec.shape[1],
            in_shard_tile_width);
    }
    if (fr_nc_in_memory_config.nd_shard_spec().has_value()) {
        TT_FATAL(
            fr_nc_full_device_grid.contains(fr_nc_in_memory_config.nd_shard_spec().value().grid),
            "FastReduceNC input ND shard grid {} must be contained in device compute grid {}",
            fr_nc_in_memory_config.nd_shard_spec().value().grid,
            fr_nc_full_device_grid);
        const auto& in_nd_shard_spec = fr_nc_in_memory_config.nd_shard_spec().value();
        if (in_nd_shard_spec.shard_shape.rank() >= 2) {
            const uint32_t in_nd_tile_height = input.tensor_spec().tile().get_height();
            const uint32_t in_nd_tile_width = input.tensor_spec().tile().get_width();
            TT_FATAL(
                in_nd_shard_spec.shard_shape[-2] > 0 && in_nd_shard_spec.shard_shape[-1] > 0,
                "FastReduceNC input ND shard last-2 dims must be positive");
            TT_FATAL(
                in_nd_shard_spec.shard_shape[-2] % in_nd_tile_height == 0,
                "FastReduceNC input ND shard_shape[-2]={} must be tile-height-aligned ({})",
                in_nd_shard_spec.shard_shape[-2],
                in_nd_tile_height);
            TT_FATAL(
                in_nd_shard_spec.shard_shape[-1] % in_nd_tile_width == 0,
                "FastReduceNC input ND shard_shape[-1]={} must be tile-width-aligned ({})",
                in_nd_shard_spec.shard_shape[-1],
                in_nd_tile_width);
        }
    }
    const auto& fr_nc_out_memory_config = args.output_mem_config;
    if (fr_nc_out_memory_config.shard_spec().has_value()) {
        TT_FATAL(
            fr_nc_full_device_grid.contains(fr_nc_out_memory_config.shard_spec().value().grid),
            "FastReduceNC output shard grid {} must be contained in device compute grid {}",
            fr_nc_out_memory_config.shard_spec().value().grid,
            fr_nc_full_device_grid);
        const auto& out_shard_spec = fr_nc_out_memory_config.shard_spec().value();
        const uint32_t out_shard_tile_height = input.tensor_spec().tile().get_height();
        const uint32_t out_shard_tile_width = input.tensor_spec().tile().get_width();
        TT_FATAL(
            out_shard_spec.shape[0] > 0 && out_shard_spec.shape[1] > 0,
            "FastReduceNC output shard_shape must be positive, got [{}, {}]",
            out_shard_spec.shape[0],
            out_shard_spec.shape[1]);
        TT_FATAL(
            out_shard_spec.shape[0] % out_shard_tile_height == 0,
            "FastReduceNC output shard_shape[0]={} must be tile-height-aligned ({})",
            out_shard_spec.shape[0],
            out_shard_tile_height);
        TT_FATAL(
            out_shard_spec.shape[1] % out_shard_tile_width == 0,
            "FastReduceNC output shard_shape[1]={} must be tile-width-aligned ({})",
            out_shard_spec.shape[1],
            out_shard_tile_width);
    }
    if (fr_nc_out_memory_config.nd_shard_spec().has_value()) {
        TT_FATAL(
            fr_nc_full_device_grid.contains(fr_nc_out_memory_config.nd_shard_spec().value().grid),
            "FastReduceNC output ND shard grid {} must be contained in device compute grid {}",
            fr_nc_out_memory_config.nd_shard_spec().value().grid,
            fr_nc_full_device_grid);
        const auto& out_nd_shard_spec = fr_nc_out_memory_config.nd_shard_spec().value();
        if (out_nd_shard_spec.shard_shape.rank() >= 2) {
            const uint32_t out_nd_tile_height = input.tensor_spec().tile().get_height();
            const uint32_t out_nd_tile_width = input.tensor_spec().tile().get_width();
            TT_FATAL(
                out_nd_shard_spec.shard_shape[-2] > 0 && out_nd_shard_spec.shard_shape[-1] > 0,
                "FastReduceNC output ND shard last-2 dims must be positive");
            TT_FATAL(
                out_nd_shard_spec.shard_shape[-2] % out_nd_tile_height == 0,
                "FastReduceNC output ND shard_shape[-2]={} must be tile-height-aligned ({})",
                out_nd_shard_spec.shard_shape[-2],
                out_nd_tile_height);
            TT_FATAL(
                out_nd_shard_spec.shard_shape[-1] % out_nd_tile_width == 0,
                "FastReduceNC output ND shard_shape[-1]={} must be tile-width-aligned ({})",
                out_nd_shard_spec.shard_shape[-1],
                out_nd_tile_width);
        }
    }
}

TensorSpec FastReduceNCDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input = tensor_args.input;
    const auto& input_shape = input.padded_shape();

    // keepdim=true
    auto output_shape = input_shape;
    // last 2-dim
    output_shape[args.dim] = 1;
    const auto output_dtype = args.output_dtype.value_or(input.dtype());
    return TensorSpec(
        output_shape,
        operations::TensorLayout(output_dtype, operations::PageConfig(Layout::TILE), args.output_mem_config));
}

Tensor FastReduceNCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fast_reduce_nc(
    const Tensor& input,
    const int32_t& dim,
    const std::optional<const Tensor>& output,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<DataType>& output_dtype) {
    using OperationType = ttnn::experimental::prim::FastReduceNCDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dim = dim,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .sub_core_grids = sub_core_grids,
        .output_dtype = output_dtype};
    auto tensor_args = OperationType::tensor_args_t{.input = input, .preallocated_output = output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
