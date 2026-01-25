// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <variant>

#include <ttnn/types.hpp>

#include <ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp>
#include <ttnn/operations/sliding_window/op_slicing/op_slicing.hpp>

namespace ttnn::operations::conv::conv_transpose2d {

using ttnn::prim::Conv2dConfig;
using ttnn::prim::Conv2dSliceConfig;

// Struct to hold all computed dimensions for transposed conv2d
// Transposed conv2d is implemented as conv2d with transformed parameters:
// - stride is always 1x1 for the conv2d micro-op
// - padding is always 0 (halo already adds padding)
// - input dimensions are "full_input" dimensions (after halo expansion)
struct ConvTranspose2dDimensions {
    // Output dimensions of the transposed convolution
    uint32_t output_height;
    uint32_t output_width;

    // Full input dimensions for the conv2d micro-op (after halo/padding expansion)
    uint32_t full_input_height;
    uint32_t full_input_width;

    // Strided input dimensions (after adding interleaved 0s for stride > 1)
    uint32_t strided_input_height;
    uint32_t strided_input_width;

    // Real padding that will be applied by halo operation
    uint32_t input_pad_top;
    uint32_t input_pad_bottom;
    uint32_t input_pad_left;
    uint32_t input_pad_right;

    // Constants for conv2d micro-op parameters (always the same for transposed conv2d)
    static constexpr std::array<uint32_t, 2> CONV2D_STRIDE = {1, 1};
    static constexpr std::array<uint32_t, 4> CONV2D_PADDING = {0, 0, 0, 0};
};

// Enum to represent the execution path for conv2d operations
enum class ConvT2dExecutionPath {
    L1,   // Execute conv2d using L1 memory
    DRAM  // Execute conv2d using DRAM slicing
};

// Helper function to determine which conv2d execution path to take based on
// slice configuration and input tensor properties
ConvT2dExecutionPath determine_conv_transpose2d_execution_path(
    const tt::tt_metal::StorageType& storage_type,
    const MemoryConfig& memory_config,
    const std::optional<const op_slicing::Op2DSliceConfig>& slice_config);

// Helper function to compute all transposed conv2d dimension transformations
// This consolidates the logic that was previously duplicated across multiple files
ConvTranspose2dDimensions compute_conv_transpose2d_dimensions(
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation);

ttnn::Tensor transform_weights_for_conv_transpose2d(const ttnn::Tensor& conv_weight_tensor, bool mirror_kernel);

ttnn::Tensor prepare_conv_transpose2d_weights(
    const ttnn::Tensor& weight_tensor,
    ttnn::MemoryConfig input_memory_config,
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
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool mirror_kernel = true);

ttnn::Tensor prepare_conv_transpose2d_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    ttnn::Layout input_layout,
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
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_);

}  // namespace ttnn::operations::conv::conv_transpose2d
