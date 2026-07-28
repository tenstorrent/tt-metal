// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe_gate_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/buffer.hpp>

#include "deepseek_moe_gate_program_descriptor_builder.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

void DeepseekMoeGateDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

void DeepseekMoeGateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    using tt::tt_metal::DataType;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& input_indices_tensor = tensor_args.input_indices_tensor;
    const auto& output_tensor = tensor_args.output_tensor;
    const auto& output_indices_tensor = tensor_args.output_indices_tensor;

    TT_FATAL(input_tensor.storage_type() == ttnn::StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(bias_tensor.storage_type() == ttnn::StorageType::DEVICE, "bias_tensor must be on device");
    TT_FATAL(
        input_indices_tensor.storage_type() == ttnn::StorageType::DEVICE, "input_indices_tensor must be on device");
    TT_FATAL(output_tensor.storage_type() == ttnn::StorageType::DEVICE, "output_tensor must be on device");
    TT_FATAL(
        output_indices_tensor.storage_type() == ttnn::StorageType::DEVICE, "output_indices_tensor must be on device");

    TT_FATAL(input_tensor.device() == bias_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == input_indices_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == output_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == output_indices_tensor.device(), "All tensors must be on the same device");

    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "input_tensor must be BFLOAT16");
    TT_FATAL(bias_tensor.dtype() == DataType::BFLOAT16, "bias_tensor must be BFLOAT16");
    TT_FATAL(input_indices_tensor.dtype() == DataType::UINT16, "input_indices_tensor must be UINT16");
    TT_FATAL(output_tensor.dtype() == DataType::BFLOAT16, "output_tensor must be BFLOAT16");
    TT_FATAL(output_indices_tensor.dtype() == DataType::UINT16, "output_indices_tensor must be UINT16");

    TT_FATAL(input_tensor.is_sharded(), "input_tensor must be sharded");
    TT_FATAL(bias_tensor.is_sharded(), "bias_tensor must be sharded");
    TT_FATAL(input_indices_tensor.is_sharded(), "input_indices_tensor must be sharded");
    TT_FATAL(output_tensor.is_sharded(), "output_tensor must be sharded");
    TT_FATAL(output_indices_tensor.is_sharded(), "output_indices_tensor must be sharded");

    const auto& in_shape = input_tensor.logical_shape();
    const auto& bias_shape = bias_tensor.logical_shape();
    const auto& out_shape = output_tensor.logical_shape();
    const auto& in_idx_shape = input_indices_tensor.logical_shape();
    const auto& out_idx_shape = output_indices_tensor.logical_shape();

    TT_FATAL(bias_shape == in_shape, "Bias and input tensors must have the same shape");
    TT_FATAL(in_idx_shape == in_shape, "Input indices and input tensors must have the same shape");
    TT_FATAL(out_idx_shape == out_shape, "Output indices and output tensors must have the same shape");

    TT_FATAL(in_shape.size() >= 2, "input_tensor must have rank >= 2");
    uint32_t h = in_shape[in_shape.size() - 2];
    uint32_t w = in_shape[in_shape.size() - 1];
    TT_FATAL(h * w == 256, "Input tensor must have 256 elements per shard (last two dims)");

    const auto& input_shard = input_tensor.shard_spec().value();
    const auto& output_shard = output_tensor.shard_spec().value();
    const auto& bias_shard = bias_tensor.memory_config().shard_spec().value();
    const auto& in_indices_shard = input_indices_tensor.memory_config().shard_spec().value();
    const auto& out_indices_shard = output_indices_tensor.memory_config().shard_spec().value();

    auto all_cores = input_shard.grid;

    TT_FATAL(input_shard.shape == bias_shard.shape, "Input and bias shard shapes must match");
    TT_FATAL(input_shard.orientation == bias_shard.orientation, "Input and bias shard orientations must match");
    TT_FATAL(bias_shard.grid.contains(all_cores), "Bias shard grid must contain input shard grid");

    TT_FATAL(input_shard.shape == in_indices_shard.shape, "Input and input-indices shard shapes must match");
    TT_FATAL(
        input_shard.orientation == in_indices_shard.orientation, "Input and input-indices orientations must match");
    TT_FATAL(in_indices_shard.grid.contains(all_cores), "Input-indices shard grid must contain input shard grid");

    TT_FATAL(output_shard.grid == out_indices_shard.grid, "Output and output-indices shard grids must match");
    TT_FATAL(output_shard.shape == out_indices_shard.shape, "Output and output-indices shard shapes must match");
    TT_FATAL(output_shard.orientation == out_indices_shard.orientation, "Output orientations must match");
    TT_FATAL(output_shard.grid.contains(all_cores), "Output shard grid must contain input compute grid");

    const auto& in_tile = input_tensor.tensor_spec().tile();
    const auto& out_tile = output_tensor.tensor_spec().tile();
    TT_FATAL(in_tile == bias_tensor.tensor_spec().tile(), "Input and bias tiles must match");
    TT_FATAL(in_tile == input_indices_tensor.tensor_spec().tile(), "Input and input-indices tiles must match");
    TT_FATAL(out_tile == output_indices_tensor.tensor_spec().tile(), "Output tiles must match");

    TT_FATAL(in_tile.get_height() == 32 && in_tile.get_width() == 32, "Input tile must be 32x32");
    TT_FATAL(out_tile.get_height() == 32 && out_tile.get_width() == 32, "Output tile must be 32x32");
    TT_FATAL(input_shard.shape[0] == 32 && input_shard.shape[1] == 32, "Input shard shape must be 32x32");
    TT_FATAL(output_shard.shape[0] == 32 && output_shard.shape[1] == 32, "Output shard shape must be 32x32");
}

DeepseekMoeGateDeviceOperation::spec_return_value_t DeepseekMoeGateDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return {
        tensor_args.output_tensor.tensor_spec(),
        tensor_args.output_indices_tensor.tensor_spec(),
    };
}

DeepseekMoeGateDeviceOperation::tensor_return_value_t DeepseekMoeGateDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return {tensor_args.output_tensor, tensor_args.output_indices_tensor};
}

std::tuple<DeepseekMoeGateDeviceOperation::operation_attributes_t, DeepseekMoeGateDeviceOperation::tensor_args_t>
DeepseekMoeGateDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& bias_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& output_tensor,
    const Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid) {
    return {
        operation_attributes_t{.eps = eps, .scaling_factor = scaling_factor, .enable_sigmoid = enable_sigmoid},
        tensor_args_t{
            .input_tensor = input_tensor,
            .bias_tensor = bias_tensor,
            .input_indices_tensor = input_indices_tensor,
            .output_tensor = output_tensor,
            .output_indices_tensor = output_indices_tensor,
        },
    };
}

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate
