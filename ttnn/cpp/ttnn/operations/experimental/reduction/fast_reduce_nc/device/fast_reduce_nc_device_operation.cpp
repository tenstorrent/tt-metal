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

    auto validate_tensor_padded_spatial = [](const Tensor& t, const char* tensor_label) {
        const uint32_t tile_height = t.tensor_spec().tile().get_height();
        const uint32_t tile_width = t.tensor_spec().tile().get_width();
        const auto& padded_shape = t.padded_shape();
        TT_FATAL(
            padded_shape.rank() >= 2,
            "FastReduceNC {} padded_shape rank {} must be at least 2",
            tensor_label,
            padded_shape.rank());
        TT_FATAL(
            padded_shape[-2] > 0 && padded_shape[-1] > 0,
            "FastReduceNC {} padded last-2 dims must be positive",
            tensor_label);
        TT_FATAL(
            padded_shape[-2] % tile_height == 0,
            "FastReduceNC {} padded height {} must be tile-height-aligned ({})",
            tensor_label,
            padded_shape[-2],
            tile_height);
        TT_FATAL(
            padded_shape[-1] % tile_width == 0,
            "FastReduceNC {} padded width {} must be tile-width-aligned ({})",
            tensor_label,
            padded_shape[-1],
            tile_width);
    };

    auto validate_shard_shape_2d = [&input](uint32_t shape0, uint32_t shape1, const char* which) {
        const uint32_t tile_height = input.tensor_spec().tile().get_height();
        const uint32_t tile_width = input.tensor_spec().tile().get_width();
        TT_FATAL(
            shape0 > 0 && shape1 > 0,
            "FastReduceNC {} shard_shape must be positive, got [{}, {}]",
            which,
            shape0,
            shape1);
        TT_FATAL(
            shape0 % tile_height == 0,
            "FastReduceNC {} shard_shape[0]={} must be tile-height-aligned ({})",
            which,
            shape0,
            tile_height);
        TT_FATAL(
            shape1 % tile_width == 0,
            "FastReduceNC {} shard_shape[1]={} must be tile-width-aligned ({})",
            which,
            shape1,
            tile_width);
    };

    auto validate_nd_shard_last_two = [&input](const Shape& shard_shape, const char* which) {
        if (shard_shape.rank() < 2) {
            return;
        }
        const uint32_t tile_height = input.tensor_spec().tile().get_height();
        const uint32_t tile_width = input.tensor_spec().tile().get_width();
        TT_FATAL(
            shard_shape[-2] > 0 && shard_shape[-1] > 0, "FastReduceNC {} ND shard last-2 dims must be positive", which);
        TT_FATAL(
            shard_shape[-2] % tile_height == 0,
            "FastReduceNC {} ND shard_shape[-2]={} must be tile-height-aligned ({})",
            which,
            shard_shape[-2],
            tile_height);
        TT_FATAL(
            shard_shape[-1] % tile_width == 0,
            "FastReduceNC {} ND shard_shape[-1]={} must be tile-width-aligned ({})",
            which,
            shard_shape[-1],
            tile_width);
    };

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
        validate_tensor_padded_spatial(preallocated_output.value(), "preallocated output");
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

    validate_tensor_padded_spatial(input, "input");

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
        validate_shard_shape_2d(in_shard_spec.shape[0], in_shard_spec.shape[1], "input");
    }
    if (fr_nc_in_memory_config.nd_shard_spec().has_value()) {
        TT_FATAL(
            fr_nc_full_device_grid.contains(fr_nc_in_memory_config.nd_shard_spec().value().grid),
            "FastReduceNC input ND shard grid {} must be contained in device compute grid {}",
            fr_nc_in_memory_config.nd_shard_spec().value().grid,
            fr_nc_full_device_grid);
        const auto& in_nd_shard_spec = fr_nc_in_memory_config.nd_shard_spec().value();
        validate_nd_shard_last_two(in_nd_shard_spec.shard_shape, "input");
    }
    const auto& fr_nc_out_memory_config = args.output_mem_config;
    if (fr_nc_out_memory_config.shard_spec().has_value()) {
        TT_FATAL(
            fr_nc_full_device_grid.contains(fr_nc_out_memory_config.shard_spec().value().grid),
            "FastReduceNC output shard grid {} must be contained in device compute grid {}",
            fr_nc_out_memory_config.shard_spec().value().grid,
            fr_nc_full_device_grid);
        const auto& out_shard_spec = fr_nc_out_memory_config.shard_spec().value();
        validate_shard_shape_2d(out_shard_spec.shape[0], out_shard_spec.shape[1], "output");
    }
    if (fr_nc_out_memory_config.nd_shard_spec().has_value()) {
        TT_FATAL(
            fr_nc_full_device_grid.contains(fr_nc_out_memory_config.nd_shard_spec().value().grid),
            "FastReduceNC output ND shard grid {} must be contained in device compute grid {}",
            fr_nc_out_memory_config.nd_shard_spec().value().grid,
            fr_nc_full_device_grid);
        const auto& out_nd_shard_spec = fr_nc_out_memory_config.nd_shard_spec().value();
        validate_nd_shard_last_two(out_nd_shard_spec.shard_shape, "output");
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
