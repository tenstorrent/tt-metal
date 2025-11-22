// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <string>
#include <array>
#include <cstdint>
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "device/conv3d_device_operation.hpp"

namespace ttnn {

namespace operations::experimental::conv3d {

namespace detail {

// Prepares conv3d weight tensor for device
// Transforms PyTorch-style [out_channels, in_channels, kD, kH, kW] weight tensor
// to the format expected by conv3d device operation
// Returns a new tensor with layout=Tile
ttnn::Tensor prepare_conv3d_weights(
    const ttnn::Tensor& weight_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    const Conv3dConfig& conv_config,
    uint32_t alignment = 16);

// Prepares conv3d bias tensor for device
// Reshapes bias to [1, -1] format and converts to tile layout
// Returns a new tensor with layout=Tile
ttnn::Tensor prepare_conv3d_bias(const ttnn::Tensor& bias_tensor, uint32_t out_channels);

}  // namespace detail
}  // namespace operations::experimental::conv3d
}  // namespace ttnn
