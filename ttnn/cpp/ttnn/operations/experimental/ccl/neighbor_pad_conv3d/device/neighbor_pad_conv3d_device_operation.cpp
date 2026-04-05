// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_conv3d_device_operation.hpp"
#include "neighbor_pad_conv3d_device_operation_types.hpp"
#include "neighbor_pad_conv3d_program_factory.hpp"
#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation_types.hpp"

#include <array>
#include <cstdint>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void NpConv3dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();

    TT_FATAL(
        input_shape.size() == 5,
        "NpConv3d: Activation tensor must have 5 dimensions (BTHWC). got {}",
        input_shape.size());
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "NpConv3d: Activation tensor must be row-major.");
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32,
        "NpConv3d: Activation tensor must be bfloat16 or float32. got {}",
        input_tensor.dtype());

    const auto& weight_tensor = tensor_args.weight_tensor;
    TT_FATAL(
        weight_tensor.dtype() == DataType::BFLOAT16 || weight_tensor.dtype() == DataType::FLOAT32,
        "NpConv3d: Weight tensor must be bfloat16 or float32. got {}",
        weight_tensor.dtype());
    TT_FATAL(weight_tensor.layout() == Layout::TILE, "NpConv3d: Weight tensor must be tile.");
    TT_FATAL(
        input_tensor.dtype() == weight_tensor.dtype(),
        "NpConv3d: Input and weight tensors must have the same dtype. got {} vs {}",
        input_tensor.dtype(),
        weight_tensor.dtype());

    if (tensor_args.bias_tensor.has_value()) {
        const auto& bias_tensor = tensor_args.bias_tensor.value();
        TT_FATAL(bias_tensor.layout() == Layout::TILE, "NpConv3d: Bias tensor must be tiled.");
        TT_FATAL(
            bias_tensor.dtype() == input_tensor.dtype(),
            "NpConv3d: Bias tensor must have the same dtype as input tensor. got {} vs {}",
            bias_tensor.dtype(),
            input_tensor.dtype());
        TT_FATAL(
            bias_tensor.logical_shape().size() == 2,
            "NpConv3d: Bias tensor must have 2 dimensions. got {}",
            bias_tensor.logical_shape().size());
    }

    TT_FATAL(
        input_shape[4] % args.groups == 0,
        "NpConv3d: Input channels must be divisible by groups. Got input channels {} and groups {}",
        input_shape[4],
        args.groups);
    TT_FATAL(
        args.output_channels % args.groups == 0,
        "NpConv3d: Output channels must be divisible by groups. Got output channels {} and groups {}",
        args.output_channels,
        args.groups);
    TT_FATAL(
        args.padding_mode == "zeros" || args.padding_mode == "replicate",
        "NpConv3d: Padding mode must be zeros or replicate. got {}",
        args.padding_mode);
    TT_FATAL(
        args.dilation[0] >= 1 && args.dilation[1] >= 1 && args.dilation[2] >= 1,
        "NpConv3d: Dilation must be >= 1 for all dimensions. got ({}, {}, {})",
        args.dilation[0],
        args.dilation[1],
        args.dilation[2]);

    // NP-specific: fabric_only + use_h_halo_buffer must be implied by fused op
    TT_FATAL(
        args.conv_config.use_h_halo_buffer,
        "NpConv3d: conv_config.use_h_halo_buffer must be true for fused NP+Conv3d op.");

    TT_FATAL(args.np_padding_h > 0, "NpConv3d: np_padding_h must be > 0 (H-halo must be needed for fused op).");

    // Validate weight shape
    const auto patch_size = args.kernel_size[0] * args.kernel_size[1] * args.kernel_size[2] * input_shape[4];
    TT_FATAL(
        weight_tensor.logical_shape()[0] == patch_size,
        "NpConv3d: Weight patch size must match input patch size. got {} vs {}",
        weight_tensor.logical_shape()[0],
        patch_size);
    TT_FATAL(
        weight_tensor.logical_shape()[1] == args.output_channels,
        "NpConv3d: Weight output channels must match. got {} vs {}",
        weight_tensor.logical_shape()[1],
        args.output_channels);
    if (tensor_args.bias_tensor.has_value()) {
        const auto& bias_tensor = tensor_args.bias_tensor.value();
        TT_FATAL(
            bias_tensor.logical_shape()[1] == args.output_channels,
            "NpConv3d: Bias must match output channels. got {} vs {}",
            bias_tensor.logical_shape()[1],
            args.output_channels);
    }

    // Validate grid size against device
    const auto& device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        args.conv_config.compute_with_storage_grid_size.x <= device_grid.x &&
            args.conv_config.compute_with_storage_grid_size.y <= device_grid.y,
        "NpConv3d: Requested grid size ({}, {}) exceeds device grid size ({}, {})",
        args.conv_config.compute_with_storage_grid_size.x,
        args.conv_config.compute_with_storage_grid_size.y,
        device_grid.x,
        device_grid.y);

    uint32_t C_in = input_shape[4];
    const uint32_t l1_alignment = hal::get_l1_alignment();
    if (args.conv_config.C_in_block > 0) {
        TT_FATAL(
            C_in % args.conv_config.C_in_block == 0,
            "NpConv3d: Input channels ({}) must be divisible by C_in_block ({})",
            C_in,
            args.conv_config.C_in_block);
        TT_FATAL(
            args.conv_config.C_in_block % l1_alignment == 0,
            "NpConv3d: C_in_block ({}) must be a multiple of {}",
            args.conv_config.C_in_block,
            l1_alignment);
    }

    if (args.conv_config.C_out_block > 0) {
        uint32_t padded_C_out = tt::round_up(args.output_channels, tt::constants::TILE_WIDTH);
        TT_FATAL(
            padded_C_out % args.conv_config.C_out_block == 0 &&
                args.conv_config.C_out_block % tt::constants::TILE_WIDTH == 0,
            "NpConv3d: C_out_block must be a multiple of {} and divide evenly into padded output channels "
            "({}). Got C_out_block={} and output_channels={}.",
            tt::constants::TILE_WIDTH,
            padded_C_out,
            args.conv_config.C_out_block,
            args.output_channels);
    }
}

TensorSpec NpConv3dDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();
    uint32_t N = input_shape[0];
    uint32_t T_in = input_shape[1];
    uint32_t H_in = input_shape[2];
    uint32_t W_in = input_shape[3];
    uint32_t C_out = args.output_channels;
    uint32_t padded_C_out = tt::round_up(C_out, tt::constants::TILE_WIDTH);

    // Inflate effective padding with halo buffer contributions so output dims are correct.
    std::array<uint32_t, 3> effective_padding = args.padding;
    if (args.conv_config.use_h_halo_buffer) {
        effective_padding[1] += args.conv_config.h_halo_padding_h;
        effective_padding[2] += args.conv_config.h_halo_padding_w;
    }

    auto [T_out, H_out, W_out] =
        detail::compute_output_dims(T_in, H_in, W_in, effective_padding, args.stride, args.kernel_size, args.dilation);

    ttnn::Shape output_shape({N, T_out, H_out, W_out, C_out});
    ttnn::Shape padded_output_shape({N, T_out, H_out, W_out, padded_C_out});

    const auto& memory_config = args.conv_output_mem_config;
    auto dtype = args.dtype;

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), memory_config, output_shape, padded_output_shape));
}

Tensor NpConv3dDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t NpConv3dDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;

    operation::Hash hash = operation::hash_operation<NpConv3dDeviceOperation>(
        args,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.logical_shape(),
        weight_tensor.dtype(),
        weight_tensor.memory_config(),
        weight_tensor.logical_shape(),
        bias_tensor.has_value());

    return hash;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor neighbor_pad_conv3d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Tensor& halo_buffer,
    const ttnn::experimental::prim::NpConv3dParams& params) {
    using OperationType = ttnn::experimental::prim::NpConv3dDeviceOperation;

    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input, .weight_tensor = weight, .bias_tensor = bias, .halo_buffer = halo_buffer};

    return ttnn::device_operation::launch<OperationType>(params, tensor_args);
}

}  // namespace ttnn::prim
