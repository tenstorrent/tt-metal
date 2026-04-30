// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include <enchantum/enchantum.hpp>
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

void AccumulationDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& optional_out{tensor_args.opt_output};
    auto out_memory_config{optional_out.has_value() ? optional_out->memory_config() : attributes.output_memory_config};

    if (optional_out.has_value()) {
        const auto& preallocated_output_shape = optional_out.value().logical_shape();
        TT_FATAL(
            input_shape == preallocated_output_shape,
            "The shapes of the input and the preallocated tensors are not equal.\n"
            "Input tensor's shape: {}\n"
            "Preallocated tensor's shape: {}",
            input_shape,
            preallocated_output_shape);
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "ttnn accumulation operations (cumprod, cumsum) require input to be on a Tenstorrent device. "
        "The input tensor is stored on {}.",
        enchantum::to_string(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "ttnn accumulation operations (cumprod, cumsum) require to be allocated in buffers on the device. "
        "The buffer is null.");

    TT_FATAL(
        !input_tensor.is_sharded(),
        "ttnn accumulation operations (cumprod, cumsum) do not support sharded input tensors.");

    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "The provided input tensor has a non-tile layout: {}.",
        enchantum::to_string(input_tensor.layout()));

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "ttnn accumulation operations (cumprod, cumsum) require the memory layout of the input tensor to be "
        "interleaved. Instead, it is {}.",
        enchantum::to_string(input_tensor.memory_config().memory_layout()));
    {
        const uint32_t input_tile_height = input_tensor.tensor_spec().tile().get_height();
        const uint32_t input_tile_width = input_tensor.tensor_spec().tile().get_width();
        const auto& padded_shape = input_tensor.padded_shape();
        TT_FATAL(
            padded_shape.rank() >= 2,
            "Accumulation input padded_shape rank {} must be at least 2 for spatial H/W checks",
            padded_shape.rank());
        TT_FATAL(
            padded_shape[-2] > 0 && padded_shape[-1] > 0,
            "Accumulation input padded spatial dims must be positive: height={}, width={}",
            padded_shape[-2],
            padded_shape[-1]);
        TT_FATAL(
            padded_shape[-2] % input_tile_height == 0,
            "Accumulation input padded_height={} must be tile-height-aligned ({})",
            padded_shape[-2],
            input_tile_height);
        TT_FATAL(
            padded_shape[-1] % input_tile_width == 0,
            "Accumulation input padded_width={} must be tile-width-aligned ({})",
            padded_shape[-1],
            input_tile_width);
        if (out_memory_config.shard_spec().has_value()) {
            const auto& output_shard_spec = out_memory_config.shard_spec().value();
            TT_FATAL(
                output_shard_spec.shape[0] > 0 && output_shard_spec.shape[1] > 0,
                "Accumulation output shard face must be positive, got shard_shape[0]={}, [1]={}",
                output_shard_spec.shape[0],
                output_shard_spec.shape[1]);
            TT_FATAL(
                output_shard_spec.shape[0] % input_tile_height == 0,
                "Accumulation output shard_shape[0]={} must be tile-height-aligned ({})",
                output_shard_spec.shape[0],
                input_tile_height);
            TT_FATAL(
                output_shard_spec.shape[1] % input_tile_width == 0,
                "Accumulation output shard_shape[1]={} must be tile-width-aligned ({})",
                output_shard_spec.shape[1],
                input_tile_width);
        }
        if (out_memory_config.nd_shard_spec().has_value()) {
            const auto& output_nd_shard_spec = out_memory_config.nd_shard_spec().value();
            if (output_nd_shard_spec.shard_shape.rank() >= 2) {
                TT_FATAL(
                    output_nd_shard_spec.shard_shape[-2] > 0 && output_nd_shard_spec.shard_shape[-1] > 0,
                    "Accumulation output ND shard last-2 dims must be positive, got [..., {}, {}]",
                    output_nd_shard_spec.shard_shape[-2],
                    output_nd_shard_spec.shard_shape[-1]);
                TT_FATAL(
                    output_nd_shard_spec.shard_shape[-2] % input_tile_height == 0,
                    "Accumulation output shard_shape[-2]={} must be tile-height-aligned ({})",
                    output_nd_shard_spec.shard_shape[-2],
                    input_tile_height);
                TT_FATAL(
                    output_nd_shard_spec.shard_shape[-1] % input_tile_width == 0,
                    "Accumulation output shard_shape[-1]={} must be tile-width-aligned ({})",
                    output_nd_shard_spec.shard_shape[-1],
                    input_tile_width);
            }
        }
        if (optional_out.has_value()) {
            const auto& preallocated_output = optional_out.value();
            const uint32_t preallocated_output_tile_height = preallocated_output.tensor_spec().tile().get_height();
            const uint32_t preallocated_output_tile_width = preallocated_output.tensor_spec().tile().get_width();
            const auto& preallocated_output_padded_shape = preallocated_output.padded_shape();
            TT_FATAL(
                preallocated_output_padded_shape.rank() >= 2,
                "Accumulation preallocated output padded_shape rank {} must be at least 2",
                preallocated_output_padded_shape.rank());
            TT_FATAL(
                preallocated_output_padded_shape[-2] > 0 && preallocated_output_padded_shape[-1] > 0,
                "Accumulation preallocated output padded spatial dims must be positive: height={}, width={}",
                preallocated_output_padded_shape[-2],
                preallocated_output_padded_shape[-1]);
            TT_FATAL(
                preallocated_output_padded_shape[-2] % preallocated_output_tile_height == 0,
                "Accumulation preallocated output padded_height={} must be tile-height-aligned ({})",
                preallocated_output_padded_shape[-2],
                preallocated_output_tile_height);
            TT_FATAL(
                preallocated_output_padded_shape[-1] % preallocated_output_tile_width == 0,
                "Accumulation preallocated output padded_width={} must be tile-width-aligned ({})",
                preallocated_output_padded_shape[-1],
                preallocated_output_tile_width);
        }
    }

    {
        const int32_t logical_rank = input_tensor.logical_shape().rank();
        const int32_t acc_dim = attributes.dim;
        TT_FATAL(acc_dim >= 0, "Accumulation (cumsum/cumprod) expects non-negative normalized dim, got {}", acc_dim);
        TT_FATAL(
            logical_rank > 0,
            "Accumulation (cumsum/cumprod) requires positive logical rank for axis {}, got rank {}",
            acc_dim,
            logical_rank);
        TT_FATAL(
            acc_dim < logical_rank, "Accumulation dim {} must be less than logical rank {}", acc_dim, logical_rank);
    }

    {
        using namespace tt::tt_metal;
        const auto device_grid_size = input_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            device_grid_size.x > 0 && device_grid_size.y > 0,
            "Device compute grid must be non-empty for accumulation (cumsum/cumprod), got ({}, {})",
            device_grid_size.x,
            device_grid_size.y);
        const CoreRangeSet device_grid =
            num_cores_to_corerangeset(device_grid_size.x * device_grid_size.y, device_grid_size, false);
        if (out_memory_config.shard_spec().has_value()) {
            const auto& output_shard_grid = out_memory_config.shard_spec().value().grid;
            TT_FATAL(
                device_grid.contains(output_shard_grid),
                "Accumulation output shard grid {} must be contained in device grid {}",
                output_shard_grid,
                device_grid);
        }
        if (out_memory_config.nd_shard_spec().has_value()) {
            const auto& output_nd_shard_grid = out_memory_config.nd_shard_spec().value().grid;
            TT_FATAL(
                device_grid.contains(output_nd_shard_grid),
                "Accumulation output ND shard grid {} must be contained in device grid {}",
                output_nd_shard_grid,
                device_grid);
        }
    }
}

AccumulationDeviceOperation::spec_return_value_t AccumulationDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        return tensor_args.opt_output->tensor_spec();
    }

    auto output_layout{Layout::TILE};
    if (attributes.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor.layout();
    }

    const DataType dtype =
        tensor_args.opt_output
            ? tensor_args.opt_output->dtype()
            : ((attributes.dtype == DataType::INVALID) ? tensor_args.input_tensor.dtype() : attributes.dtype);

    const auto output_shape{tensor_args.input_tensor.logical_shape()};
    return TensorSpec{output_shape, TensorLayout{dtype, output_layout, attributes.output_memory_config}};
}

AccumulationDeviceOperation::tensor_return_value_t AccumulationDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        // a copy of a Python object (referencing to the same tensor though) is returned here
        return *tensor_args.opt_output;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

operation::Hash AccumulationDeviceOperation::compute_program_hash(
    const operation_attributes_t& op_args, const tensor_args_t& tensor_args) {
    return operation::hash_operation<AccumulationDeviceOperation>(
        op_args.dim,
        op_args.output_memory_config,
        op_args.flip,
        op_args.dtype,
        op_args.op,
        tensor_args.input_tensor.logical_shape(),
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.opt_output.has_value() ? tensor_args.opt_output.value().logical_shape() : Shape{},
        tensor_args.opt_output.has_value() ? tensor_args.opt_output.value().memory_config() : MemoryConfig{},
        tensor_args.opt_output.has_value() ? tensor_args.opt_output.value().dtype() : DataType{});
}

ttnn::Tensor accumulation(
    const Tensor& input_tensor,
    const int32_t& dim,
    const std::optional<DataType>& dtype,
    const bool& reverse_order,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config,
    AccumulationOp op) {
    using OperationType = AccumulationDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            (dim < 0) ? (dim + input_tensor.logical_shape().rank()) : dim,
            dtype.has_value() ? dtype.value()
                              : (optional_out.has_value() ? optional_out->dtype() : input_tensor.dtype()),
            memory_config.has_value()
                ? *memory_config
                : (optional_out.has_value() ? optional_out->memory_config() : input_tensor.memory_config()),
            reverse_order,
            op},
        OperationType::tensor_args_t{input_tensor, std::move(optional_out)});
}

}  // namespace ttnn::prim
