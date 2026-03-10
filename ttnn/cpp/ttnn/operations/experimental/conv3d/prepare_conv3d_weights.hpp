// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::conv3d {

// Prepares conv3d weight tensor for device
// Transforms PyTorch-style [out_channels, in_channels, kD, kH, kW] weight tensor
// to the format expected by conv3d device operation.
// If device is provided, moves the tensor to device before transforms (device-side).
// If device is nullopt, transforms run on host.
// Returns a new tensor with layout=Tile.
ttnn::Tensor prepare_conv3d_weights(
    const ttnn::Tensor& weight_tensor,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t C_in_block = 0,
    uint32_t alignment = 16,
    std::optional<MeshDevice*> device = std::nullopt);

// Prepares conv3d bias tensor for device
// Reshapes bias to [1, out_channels] format and converts to tile layout.
// If device is provided, moves the tensor to device before transforms (device-side).
// If device is nullopt, transforms run on host.
// Returns a new tensor with layout=Tile.
ttnn::Tensor prepare_conv3d_bias(
    const ttnn::Tensor& bias_tensor, uint32_t out_channels, std::optional<MeshDevice*> device = std::nullopt);

}  // namespace ttnn::operations::experimental::conv3d
