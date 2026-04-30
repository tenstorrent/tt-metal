// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ema_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

#include <cmath>

namespace ttnn::prim {

using namespace tt::tt_metal;

void EmaDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    // Dtype, Device and layout checks
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", input_tensor.dtype());
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device, got: {}",
        input_tensor.storage_type());
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in a device buffer");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", input_tensor.layout());

    // Shape constraints: [1, B, C, T]
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "EMA input must be 4D [1, B, C, T], got rank {}", input_shape.rank());
    TT_FATAL(input_shape[0] == 1, "EMA expects leading dimension to be 1, got {}", input_shape[0]);

    // This OP produces as many elements in output as there are in input
    // Thus, the volume must be the same to avoid writing outside the output buffer
    if (tensor_args.optional_output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.optional_output_tensor.value();
        TT_FATAL(
            output_tensor.dtype() == DataType::BFLOAT16,
            "Output tensor must be BFLOAT16, got: {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Output tensor must be on device, got: {}",
            output_tensor.storage_type());
        TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor must be allocated in a device buffer");
        TT_FATAL(
            output_tensor.layout() == Layout::TILE,
            "Output tensor must have TILE layout, got: {}",
            output_tensor.layout());
        TT_FATAL(
            input_tensor.padded_shape().volume() == output_tensor.padded_shape().volume(),
            "Input and output must have the same volume, input: {}, output: {}",
            input_tensor.padded_shape().volume(),
            output_tensor.padded_shape().volume());
    }

    // Alpha validation
    TT_FATAL(!std::isnan(operation_attributes.alpha), "EMA alpha must be a valid number, got NaN");
    {
        const uint32_t ema_in_tile_height = input_tensor.tensor_spec().tile().get_height();
        const uint32_t ema_in_tile_width = input_tensor.tensor_spec().tile().get_width();
        const auto& ema_in_padded_shape = input_tensor.padded_shape();
        TT_FATAL(
            ema_in_padded_shape.rank() >= 2,
            "EMA input padded_shape rank {} must be at least 2 for H/W tile checks",
            ema_in_padded_shape.rank());
        TT_FATAL(
            ema_in_padded_shape[-2] > 0 && ema_in_padded_shape[-1] > 0,
            "EMA input padded spatial dims must be positive: height={}, width={}",
            ema_in_padded_shape[-2],
            ema_in_padded_shape[-1]);
        TT_FATAL(
            ema_in_padded_shape[-2] % ema_in_tile_height == 0,
            "EMA input padded_height={} must be tile-height-aligned ({})",
            ema_in_padded_shape[-2],
            ema_in_tile_height);
        TT_FATAL(
            ema_in_padded_shape[-1] % ema_in_tile_width == 0,
            "EMA input padded_width={} must be tile-width-aligned ({})",
            ema_in_padded_shape[-1],
            ema_in_tile_width);

        const auto& ema_out_mem = operation_attributes.output_mem_config;
        if (ema_out_mem.shard_spec().has_value()) {
            const auto& ema_out_shard = ema_out_mem.shard_spec().value();
            TT_FATAL(
                ema_out_shard.shape[0] > 0 && ema_out_shard.shape[1] > 0,
                "EMA output shard_shape must be positive, got [{}, {}]",
                ema_out_shard.shape[0],
                ema_out_shard.shape[1]);
            TT_FATAL(
                ema_out_shard.shape[0] % ema_in_tile_height == 0,
                "EMA output shard_shape[0]={} must be tile-height-aligned ({})",
                ema_out_shard.shape[0],
                ema_in_tile_height);
            TT_FATAL(
                ema_out_shard.shape[1] % ema_in_tile_width == 0,
                "EMA output shard_shape[1]={} must be tile-width-aligned ({})",
                ema_out_shard.shape[1],
                ema_in_tile_width);
        }
        if (ema_out_mem.nd_shard_spec().has_value()) {
            const auto& ema_nd = ema_out_mem.nd_shard_spec().value();
            if (ema_nd.shard_shape.rank() >= 2) {
                TT_FATAL(
                    ema_nd.shard_shape[-2] > 0 && ema_nd.shard_shape[-1] > 0,
                    "EMA output ND shard last two dims must be positive");
                TT_FATAL(
                    ema_nd.shard_shape[-2] % ema_in_tile_height == 0,
                    "EMA output ND shard_shape[-2]={} must be tile-height-aligned ({})",
                    ema_nd.shard_shape[-2],
                    ema_in_tile_height);
                TT_FATAL(
                    ema_nd.shard_shape[-1] % ema_in_tile_width == 0,
                    "EMA output ND shard_shape[-1]={} must be tile-width-aligned ({})",
                    ema_nd.shard_shape[-1],
                    ema_in_tile_width);
            }
        }

        const auto ema_device_grid_sz = input_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            ema_device_grid_sz.x > 0 && ema_device_grid_sz.y > 0,
            "EMA requires non-empty device compute grid, got ({}, {})",
            ema_device_grid_sz.x,
            ema_device_grid_sz.y);
        const CoreRangeSet ema_full_device_grid =
            num_cores_to_corerangeset(ema_device_grid_sz.x * ema_device_grid_sz.y, ema_device_grid_sz, false);
        if (ema_out_mem.shard_spec().has_value()) {
            TT_FATAL(
                ema_full_device_grid.contains(ema_out_mem.shard_spec().value().grid),
                "EMA output shard grid {} must be contained in device compute grid {}",
                ema_out_mem.shard_spec().value().grid,
                ema_full_device_grid);
        }
        if (ema_out_mem.nd_shard_spec().has_value()) {
            TT_FATAL(
                ema_full_device_grid.contains(ema_out_mem.nd_shard_spec().value().grid),
                "EMA output ND shard grid {} must be contained in device compute grid {}",
                ema_out_mem.nd_shard_spec().value().grid,
                ema_full_device_grid);
        }

        if (tensor_args.optional_output_tensor.has_value()) {
            const auto& prealloc_out = tensor_args.optional_output_tensor.value();
            const uint32_t prealloc_out_tile_h = prealloc_out.tensor_spec().tile().get_height();
            const uint32_t prealloc_out_tile_w = prealloc_out.tensor_spec().tile().get_width();
            const auto& prealloc_out_padded = prealloc_out.padded_shape();
            TT_FATAL(
                prealloc_out_padded.rank() >= 2,
                "EMA preallocated output padded_shape rank {} must be at least 2",
                prealloc_out_padded.rank());
            TT_FATAL(
                prealloc_out_padded[-2] > 0 && prealloc_out_padded[-1] > 0,
                "EMA preallocated output padded spatial dims must be positive");
            TT_FATAL(
                prealloc_out_padded[-2] % prealloc_out_tile_h == 0,
                "EMA preallocated output padded_height={} must be tile-height-aligned ({})",
                prealloc_out_padded[-2],
                prealloc_out_tile_h);
            TT_FATAL(
                prealloc_out_padded[-1] % prealloc_out_tile_w == 0,
                "EMA preallocated output padded_width={} must be tile-width-aligned ({})",
                prealloc_out_padded[-1],
                prealloc_out_tile_w);
        }
    }
}

TensorSpec EmaDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    return tensor_args.input.tensor_spec().with_memory_config(operation_attributes.output_mem_config);
}

Tensor EmaDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ttnn::Tensor ema_device(
    const Tensor& input,
    float alpha,
    CoreCoord grid_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = EmaDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .alpha = alpha,
            .grid_size = grid_size,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        OperationType::tensor_args_t{
            .input = input,
            .optional_output_tensor = std::move(optional_output_tensor),
        });
}

}  // namespace ttnn::prim
