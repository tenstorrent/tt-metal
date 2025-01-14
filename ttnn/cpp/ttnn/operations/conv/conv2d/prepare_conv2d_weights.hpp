
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::sliding_window {
struct ParallelConfig;
}

namespace ttnn {

namespace operations::conv {
namespace conv2d {

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to tilized 2d matrix layout for block sharded conv. Adds zero padding between weight
// blocks based on output shard width padding. Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout_block_sharded(
    const Tensor& conv_weight_tensor, uint32_t num_channel_shards, std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution bias to tilized layout for block sharded conv. Adds zero padding between bias blocks based on
// output shard width padding. Returns a new tensor with layout=Tile
Tensor convert_conv_bias_tensor_to_tiled_layout_block_sharded(
    const Tensor& conv_bias_tensor, uint32_t num_channel_shards, std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to tilized 2d matrix layout with special block height padding
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to grouped layout with padded zeros
Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype);

// Converts convolution weights to depthwise layout with broadcasted weights
Tensor convert_conv_weight_tensor_to_depthwise_layout(
    const Tensor& conv_weight_tensor, uint32_t act_block_h_ntiles, DataType output_dtype);

template <typename T>
ttnn::Tensor conv_bias_layout_convert(
    const ttnn::Tensor& bias_tensor,
    DataType bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const sliding_window::ParallelConfig& parallel_config,
    T* device,
    uint32_t out_channels,
    bool is_non_tile_mul_width);

template <typename T>
ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    T* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template <typename T>
ttnn::Tensor prepare_conv_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_tensor_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    T* device,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

template <typename T>
std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const sliding_window::ParallelConfig& input_parallel_config,
    const sliding_window::ParallelConfig& output_parallel_config,
    T* device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width,
    const bool parameters_on_device = true,
    bool is_non_tile_mul_width = false);

}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
