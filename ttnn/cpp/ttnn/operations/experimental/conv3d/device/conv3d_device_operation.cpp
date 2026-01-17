// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_device_operation.hpp"
#include "conv3d_device_operation_types.hpp"
#include "conv3d_program_factory.hpp"
#include <array>
#include <cstdint>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::conv3d {

namespace detail {
std::tuple<uint32_t, uint32_t, uint32_t> compute_output_dims(
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& kernel_size) {
    uint32_t T_out = ((T_in + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1;
    uint32_t H_out = ((H_in + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1;
    uint32_t W_out = ((W_in + 2 * padding[2] - kernel_size[2]) / stride[2]) + 1;
    return {T_out, H_out, W_out};
}
}  // namespace detail

Conv3dDeviceOperation::program_factory_t Conv3dDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::Conv3dProgramFactory{};
}

void Conv3dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void Conv3dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;

    TT_FATAL(
        input_tensor_a.logical_shape().size() == 5,
        "Activation tensor must have 5 dimensions. got {}",
        input_tensor_a.logical_shape().size());
    // check row-major
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Activation tensor must be row-major.");

    // input and weight must both be interleaved, bfloat16
    TT_FATAL(!input_tensor_a.memory_config().is_sharded(), "Activation tensor must be interleaved.");
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Activation tensor must be bfloat16.");

    const auto& weight_tensor = tensor_args.weight_tensor;
    TT_FATAL(!weight_tensor.memory_config().is_sharded(), "Weight tensor must be interleaved.");
    TT_FATAL(weight_tensor.dtype() == DataType::BFLOAT16, "Weight tensor must be bfloat16.");
    TT_FATAL(weight_tensor.layout() == Layout::TILE, "Weight tensor must be tile.");

    if (tensor_args.bias_tensor.has_value()) {
        const auto& bias_tensor = tensor_args.bias_tensor.value();
        TT_FATAL(!bias_tensor.memory_config().is_sharded(), "Bias tensor must be interleaved.");
        TT_FATAL(bias_tensor.layout() == Layout::TILE, "Bias tensor must be tiled.");
        TT_FATAL(
            bias_tensor.dtype() == DataType::BFLOAT16, "Bias tensor must be bfloat16. got {}", bias_tensor.dtype());
        TT_FATAL(
            bias_tensor.logical_shape().size() == 2,
            "Bias tensor must have 2 dimensions. got {}",
            bias_tensor.logical_shape().size());
    }

    TT_FATAL(args.groups == 1, "Groups must be 1. got {}", args.groups);
    // assert padding on T is zero
    TT_FATAL(
        args.padding[0] == 0,
        "Padding must be (0,x,x). got ({}, {}, {})",
        args.padding[0],
        args.padding[1],
        args.padding[2]);
    TT_FATAL(
        args.padding_mode == "zeros" || args.padding_mode == "replicate",
        "Padding mode must be zeros or replicate. got {}",
        args.padding_mode);

    if (args.config.C_out_block > 0) {
        TT_FATAL(
            args.output_channels % args.config.C_out_block == 0 &&
                args.config.C_out_block % tt::constants::TILE_WIDTH == 0,
            "C_out_block must be a multiple of {} and divide evenly into output channels. Got C_out_block={} and "
            "output_channels={}.",
            tt::constants::TILE_WIDTH,
            args.config.C_out_block,
            args.output_channels);
    }

    TT_FATAL(
        args.output_channels % tt::constants::TILE_WIDTH == 0,
        "Output channels must be a multiple of {}.",
        tt::constants::TILE_WIDTH);

    // Validate weight shape and config arguments
    const auto patch_size =
        args.kernel_size[0] * args.kernel_size[1] * args.kernel_size[2] * input_tensor_a.logical_shape()[4];
    TT_FATAL(
        weight_tensor.logical_shape()[0] == patch_size,
        "Weight patch size must match input patch size. got {} vs {}",
        weight_tensor.logical_shape()[0],
        patch_size);
    TT_FATAL(
        weight_tensor.logical_shape()[1] == args.output_channels,
        "Weight output channels must match input output channels. got {} vs {}",
        weight_tensor.logical_shape()[1],
        args.output_channels);
    if (tensor_args.bias_tensor.has_value()) {
        const auto& bias_tensor = tensor_args.bias_tensor.value();
        TT_FATAL(
            bias_tensor.logical_shape()[1] == args.output_channels,
            "Bias must match output channels. got {} vs {}",
            bias_tensor.logical_shape()[1],
            args.output_channels);
    }

    // Add grid size validation
    const auto& device_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_FATAL(
        args.config.compute_with_storage_grid_size.x <= device_grid.x &&
            args.config.compute_with_storage_grid_size.y <= device_grid.y,
        "Requested grid size ({}, {}) exceeds device grid size ({}, {})",
        args.config.compute_with_storage_grid_size.x,
        args.config.compute_with_storage_grid_size.y,
        device_grid.x,
        device_grid.y);

    uint32_t C_in = input_tensor_a.logical_shape()[4];
    const uint32_t l1_alignment = hal::get_l1_alignment();
    if (args.config.C_in_block > 0) {
        TT_FATAL(
            C_in % args.config.C_in_block == 0,
            "Input channels ({}) must be divisible by C_in_block ({})",
            C_in,
            args.config.C_in_block);
        TT_FATAL(
            args.config.C_in_block % l1_alignment == 0,
            "C_in_block ({}) must be a multiple of {}",
            args.config.C_in_block,
            l1_alignment);
    }

    // Verify number of C_in_blocks is <= the number of cores
    uint32_t C_in_block = (args.config.C_in_block > 0) ? args.config.C_in_block : C_in;
    uint32_t C_in_blocks = C_in / C_in_block;
    uint32_t total_cores = args.config.compute_with_storage_grid_size.x * args.config.compute_with_storage_grid_size.y;
    TT_FATAL(
        C_in_blocks <= total_cores,
        "Number of C_in blocks ({}) must be <= the number of cores ({})",
        C_in_blocks,
        total_cores);
}

TensorSpec Conv3dDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;
    const auto& input_tensor_a_shape = input_tensor_a.logical_shape();
    uint32_t N = input_tensor_a_shape[0];
    uint32_t T_in = input_tensor_a_shape[1];
    uint32_t H_in = input_tensor_a_shape[2];
    uint32_t W_in = input_tensor_a_shape[3];
    uint32_t C_out = args.output_channels;

    auto [T_out, H_out, W_out] =
        detail::compute_output_dims(T_in, H_in, W_in, args.padding, args.stride, args.kernel_size);

    ttnn::Shape output_shape({N, T_out, H_out, W_out, C_out});

    const auto& memory_config = args.output_mem_config;
    auto dtype = args.dtype;

    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), memory_config));
}

Tensor Conv3dDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t Conv3dDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.padded_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<Conv3dDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_tensor.memory_config(), input_shape.volume());

    return hash;
}

}  // namespace ttnn::operations::experimental::conv3d

namespace ttnn::prim {

ttnn::operations::experimental::conv3d::Conv3dDeviceOperation::tensor_return_value_t conv3d(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<Tensor>& bias_tensor,
    const ttnn::operations::experimental::conv3d::Conv3dConfig& config,
    tt::tt_metal::DataType dtype_,
    uint32_t output_channels_,
    const std::array<uint32_t, 3>& kernel_size_,
    const std::array<uint32_t, 3>& stride_,
    const std::array<uint32_t, 3>& padding_,
    const std::array<uint32_t, 3>& dilation_,
    const std::string& padding_mode_,
    uint32_t groups_,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::conv3d::Conv3dDeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    auto operation_attributes = OperationType::operation_attributes_t{
        .config = config,
        .output_mem_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        .compute_kernel_config = kernel_config_val,
        .dtype = dtype_,
        .output_channels = output_channels_,
        .kernel_size = kernel_size_,
        .stride = stride_,
        .padding = padding_,
        .dilation = dilation_,
        .padding_mode = padding_mode_,
        .groups = groups_};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor, .weight_tensor = weight_tensor, .bias_tensor = bias_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
