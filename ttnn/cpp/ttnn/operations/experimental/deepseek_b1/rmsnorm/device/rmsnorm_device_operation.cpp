// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::deepseek_b1::rmsnorm {

RmsnormDeviceOperation::program_factory_t RmsnormDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RmsnormProgramFactory{};
}

void RmsnormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& gamma_tensor = tensor_args.gamma_tensor;
    const auto& output_tensor = tensor_args.output_tensor;

    // Validate input tensor
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffer on device!");
    TT_FATAL(input_tensor.memory_config().is_sharded(), "Input tensor must be sharded");
    TT_FATAL(input_tensor.memory_config().shard_spec().has_value(), "Input tensor must have a shard spec");

    const auto& input_shard_spec = *input_tensor.memory_config().shard_spec();

    // Validate that input is sharded on exactly one core
    TT_FATAL(
        input_shard_spec.grid.num_cores() == 1,
        "Input tensor must be sharded on exactly one core, but found {} cores",
        input_shard_spec.grid.num_cores());

    // Validate input shape is [1, N]
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(input_shape.rank() == 2, "Input tensor must be rank 2, but got rank {}", input_shape.rank());
    TT_FATAL(
        input_shape[0] == 1, "Input tensor shape must be [1, N], but got [{}, {}]", input_shape[0], input_shape[1]);

    // Validate that N is a multiple of TILE_WIDTH
    TT_FATAL(
        input_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input width must be a multiple of TILE_WIDTH ({}), but got {}",
        tt::constants::TILE_WIDTH,
        input_shape[1]);

    uint32_t num_sticks = input_shape[1] / tt::constants::TILE_WIDTH;
    bool tiny_tile = (input_tensor.logical_shape()[1] / tt::constants::TILE_WIDTH) % tt::constants::TILE_HEIGHT != 0;
    if (tiny_tile) {
        TT_FATAL(num_sticks % (tt::constants::TILE_HEIGHT / 2) == 0, "Input must be packable into tiles");
    }
    // TODO: This disables tile tiles. Figure out why tiny tiles hang and remove this check.
    // TT_FATAL(num_sticks % tt::constants::TILE_HEIGHT == 0, "Input must be packable into tiles");

    TT_FATAL(
        operation_attributes.numel > 0 && operation_attributes.numel <= input_tensor.logical_volume(),
        "Numel must be greater than 0 and less than or equal to input tensor logical volume");

    // Validate input dtype is bfloat16
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16, "Input tensor must be bfloat16, but got {}", input_tensor.dtype());

    // Lambda to validate that a tensor matches input tensor's properties
    auto validate_tensor_matches_input = [&](const Tensor& tensor, const std::string& tensor_name) {
        // Validate basic tensor properties
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", tensor_name);
        TT_FATAL(tensor.buffer() != nullptr, "{} tensor must be allocated in buffer on device!", tensor_name);
        TT_FATAL(tensor.layout() == input_tensor.layout(), "{} must have the same layout as input", tensor_name);
        TT_FATAL(tensor.memory_config().is_sharded(), "{} tensor must be sharded", tensor_name);
        TT_FATAL(tensor.memory_config().shard_spec().has_value(), "{} tensor must have a shard spec", tensor_name);

        // Validate dtype matches input
        TT_FATAL(
            tensor.dtype() == input_tensor.dtype(),
            "{} dtype must match input dtype. Input: {}, {}: {}",
            tensor_name,
            input_tensor.dtype(),
            tensor_name,
            tensor.dtype());

        // Validate shape matches input
        const auto& tensor_shape = tensor.logical_shape();
        TT_FATAL(
            input_shape == tensor_shape,
            "{} shape must match input shape. Input: [{}, {}], {}: [{}, {}]",
            tensor_name,
            input_shape[0],
            input_shape[1],
            tensor_name,
            tensor_shape[0],
            tensor_shape[1]);

        // Validate shard spec matches input
        const auto& tensor_shard_spec = *tensor.memory_config().shard_spec();
        TT_FATAL(
            input_shard_spec.grid == tensor_shard_spec.grid,
            "{} must be sharded on the same core as input",
            tensor_name);
        TT_FATAL(
            input_shard_spec.shape == tensor_shard_spec.shape,
            "{} shard spec must match input shard spec. Input: [{}, {}], {}: [{}, {}]",
            tensor_name,
            input_shard_spec.shape[0],
            input_shard_spec.shape[1],
            tensor_name,
            tensor_shard_spec.shape[0],
            tensor_shard_spec.shape[1]);
        TT_FATAL(
            input_shard_spec.orientation == tensor_shard_spec.orientation,
            "{} shard orientation must match input shard orientation",
            tensor_name);
    };

    // Validate gamma and output tensors match input
    validate_tensor_matches_input(gamma_tensor, "Gamma");
    validate_tensor_matches_input(output_tensor, "Output");

    uint32_t tile_height = tiny_tile ? tt::constants::TILE_HEIGHT / 2 : tt::constants::TILE_HEIGHT;
    tt::tt_metal::Tile tile({tile_height, tt::constants::TILE_WIDTH});
    uint32_t dest_reg_count = get_dest_reg_count(operation_attributes.compute_kernel_config, tile.get_tile_shape());
    uint32_t num_tiles = input_tensor.logical_volume() / (tile.get_tile_hw());
    // This is - 1 in prep for #32998, where we use 1 tile in dst for the RMS value
    TT_FATAL(
        num_tiles <= (dest_reg_count - 1),
        "Number of tiles {} must be less than or equal to dest register count {}",
        num_tiles,
        dest_reg_count - 1);
}

void RmsnormDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

RmsnormDeviceOperation::spec_return_value_t RmsnormDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

RmsnormDeviceOperation::tensor_return_value_t RmsnormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Return the pre-allocated output tensor
    return tensor_args.output_tensor;
}

std::tuple<RmsnormDeviceOperation::operation_attributes_t, RmsnormDeviceOperation::tensor_args_t>
RmsnormDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& gamma_tensor,
    const Tensor& output_tensor,
    float epsilon,
    uint32_t numel,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    auto arch = input_tensor.device()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);

    return {
        operation_attributes_t{.epsilon = epsilon, .numel = numel, .compute_kernel_config = kernel_config_val},
        tensor_args_t{.input_tensor = input_tensor, .gamma_tensor = gamma_tensor, .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::deepseek_b1::rmsnorm
