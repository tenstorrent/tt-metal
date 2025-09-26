// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <string>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::sliding_window {
struct ParallelConfig;
}

namespace ttnn {

namespace operations::conv {
namespace conv2d {

// Device validation functions for conv2d tensors (after preparation/on device)
bool is_valid_device_conv_weights(
    const ttnn::Tensor& weight_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    const std::optional<DataType>& expected_dtype);

bool is_valid_device_conv_bias(
    const ttnn::Tensor& bias_tensor, uint32_t out_channels, const std::optional<DataType>& expected_dtype);

// Converts convolution weights to interleaved MM layout [1, 1, KhKwCi, Co] and tilizes
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_interleaved_mm_layout(
    const Tensor& conv_weight_tensor, std::optional<DataType> output_dtype = std::nullopt);

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
    const Tensor& conv_weight_tensor,
    uint32_t in_num_channel_shards,
    uint32_t out_num_channel_shards,
    bool full_inner_dim,
    std::optional<DataType> output_dtype = std::nullopt);

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
    bool enable_activation_reuse = false,
    std::optional<DataType> output_dtype = std::nullopt);

// Converts convolution weights to grouped layout with padded zeros
Tensor convert_conv_weight_tensor_to_grouped_layout(
    const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype);

// Converts convolution weights to depthwise layout with broadcasted weights
Tensor convert_conv_weight_tensor_to_depthwise_layout(
    const Tensor& conv_weight_tensor, uint32_t act_block_h_ntiles, DataType output_dtype);

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor& weight_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    bool has_bias,
    uint32_t groups,
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_);

ttnn::Tensor prepare_conv_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

// Unified parameter struct for conv2d weight and bias preparation
struct Conv2dWeightsBiasPrepConfig {
    // Constructor to ensure all required parameters are initialized
    Conv2dWeightsBiasPrepConfig(
        uint32_t input_channels_alignment_,
        std::optional<DataType> weights_bias_dtype_,
        uint32_t weight_block_h_ntiles_,
        uint32_t weight_block_w_ntiles_,
        const sliding_window::ParallelConfig& input_parallel_config_,
        const sliding_window::ParallelConfig& output_parallel_config_,
        uint32_t groups_,
        uint32_t act_block_h_ntiles_,
        uint32_t input_width_,
        bool interlaved_mm_conv,
        bool has_bias_ = false,
        bool parameters_on_device_ = true,
        bool enable_kernel_stride_folding_ = false,
        bool full_inner_dim_ = false,
        bool enable_activation_reuse_ = false,
        std::array<uint32_t, 2> kernel_size_ = {1, 1},
        std::array<uint32_t, 2> stride_ = {1, 1},
        std::array<uint32_t, 4> padding_n4_ = {0, 0, 0, 0}) :
        input_channels_alignment(input_channels_alignment_),
        weights_bias_dtype(weights_bias_dtype_),
        weight_block_h_ntiles(weight_block_h_ntiles_),
        weight_block_w_ntiles(weight_block_w_ntiles_),
        input_parallel_config(input_parallel_config_),
        output_parallel_config(output_parallel_config_),
        groups(groups_),
        act_block_h_ntiles(act_block_h_ntiles_),
        input_width(input_width_),
        has_bias(has_bias_),
        parameters_on_device(parameters_on_device_),
        enable_kernel_stride_folding(enable_kernel_stride_folding_),
        full_inner_dim(full_inner_dim_),
        enable_activation_reuse(enable_activation_reuse_),
        kernel_size(kernel_size_),
        stride(stride_),
        padding_n4(padding_n4_),
        interleaved_mm_conv(interlaved_mm_conv) {}

    // Common parameters
    const uint32_t input_channels_alignment;
    const std::optional<DataType> weights_bias_dtype;
    uint32_t weight_block_h_ntiles;
    const uint32_t weight_block_w_ntiles;
    const sliding_window::ParallelConfig input_parallel_config;
    const sliding_window::ParallelConfig output_parallel_config;
    const uint32_t groups;
    const uint32_t act_block_h_ntiles;
    const uint32_t input_width;
    const bool has_bias;
    const bool parameters_on_device;

    const bool enable_kernel_stride_folding;
    const bool full_inner_dim;
    const bool enable_activation_reuse;

    // Kernel stride folding parameters
    const std::array<uint32_t, 2> kernel_size;
    const std::array<uint32_t, 2> stride;
    const std::array<uint32_t, 4> padding_n4;
    // This conv will go through auto shard codepath for matmul based convs
    const bool interleaved_mm_conv;

    static constexpr auto attribute_names = std::make_tuple(
        "input_channels_alignment",
        "weights_bias_dtype",
        "weight_block_h_ntiles",
        "weight_block_w_ntiles",
        "input_parallel_config",
        "output_parallel_config",
        "groups",
        "act_block_h_ntiles",
        "input_width",
        "has_bias",
        "parameters_on_device",
        "enable_kernel_stride_folding",
        "full_inner_dim",
        "kernel_size",
        "stride",
        "padding_n4",
        "interleaved_mm_conv");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->input_channels_alignment),
            std::cref(this->weights_bias_dtype),
            std::cref(this->weight_block_h_ntiles),
            std::cref(this->weight_block_w_ntiles),
            std::cref(this->input_parallel_config),
            std::cref(this->output_parallel_config),
            std::cref(this->groups),
            std::cref(this->act_block_h_ntiles),
            std::cref(this->input_width),
            std::cref(this->has_bias),
            std::cref(this->parameters_on_device),
            std::cref(this->enable_kernel_stride_folding),
            std::cref(this->full_inner_dim),
            std::cref(this->kernel_size),
            std::cref(this->stride),
            std::cref(this->padding_n4),
            std::cref(this->interleaved_mm_conv));
    }
};

std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    Conv2dWeightsBiasPrepConfig& params,
    MeshDevice* device);

std::optional<ttnn::Tensor> prepare_conv_bias_internal(
    const std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t out_channels,
    const Conv2dWeightsBiasPrepConfig& params,
    DataType weight_dtype,
    MeshDevice* device);
}  // namespace conv2d
}  // namespace operations::conv
}  // namespace ttnn
