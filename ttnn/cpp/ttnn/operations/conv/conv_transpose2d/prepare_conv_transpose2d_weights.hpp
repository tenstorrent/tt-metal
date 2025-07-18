// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"

namespace ttnn {

namespace operations::conv {
namespace conv_transpose2d {

Tensor transform_weights_for_conv_transpose2d(const Tensor& conv_weight_tensor, bool mirror_kernel);

template <typename T>
ttnn::Tensor prepare_conv_transpose2d_weights(
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
    T* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    bool mirror_kernel = true);

template <typename T>
ttnn::Tensor prepare_conv_transpose2d_bias(
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
    T* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_);

}  // namespace conv_transpose2d
}  // namespace operations::conv
}  // namespace ttnn
