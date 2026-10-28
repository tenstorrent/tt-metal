// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "grid_sample_prepare_grid.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations::grid_sample {

/**
 * Grid sample operation for spatial sampling (bilinear or nearest interpolation).
 *
 * Samples the input tensor at the grid locations. Grid coordinates are normalized to the
 * [-1, 1] range; coordinates outside the image produce zeros. All tensors are channels-last
 * and row-major: each sample reads one pixel's channels as a single row-major stick.
 *
 * Args:
 *   input_tensor: Input tensor of shape (N, H_in, W_in, C), row-major, interleaved or
 *     height-sharded.
 *   grid: Sampling grid with the coordinates along the last dimension, x first, y second:
 *     (N, H_out, W_out, 2) for one point per output location, or (N, H_out, W_out, 2K) for K
 *     points per location (batched grid). In nearest mode the batched grid may also be given
 *     unmerged as (N, H_out, W_out, K, 2) (bf16, interleaved, batch_output_channels=true,
 *     height-sharded output): the kernel then reads the K pairs of a location itself, which
 *     avoids a row-major reshape on device.
 *   mode: Interpolation mode, "bilinear" or "nearest"
 *   padding_mode: How to handle out-of-bounds coordinates, currently only "zeros" is supported
 *   align_corners: Whether to align corners when mapping normalized coordinates to pixel indices
 *   use_precomputed_grid: Whether the grid already holds pixel indices and weights (see
 *     grid_sample_prepare_grid) instead of normalized coordinates
 *   batch_output_channels: Where the K samples of a batched grid go in the output: along the
 *     channels (true) or along the width (false)
 *   memory_config: Memory configuration for the output tensor
 *   compute_kernel_config: Optional compute kernel configuration (math fidelity, fp32 dest
 *     accumulation, etc.). Defaults are arch-specific.
 *
 * Returns:
 *   Output tensor, row-major, channels-last: (N, H_out, W_out, C) for a single point per
 *   location; for a batched grid (N, H_out, W_out, K * C) with batch_output_channels=true or
 *   (N, H_out, W_out * K, C) with batch_output_channels=false.
 */
ttnn::Tensor grid_sample(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& grid,
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool align_corners = false,
    bool use_precomputed_grid = false,
    bool batch_output_channels = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace operations::grid_sample

// Import functions to main ttnn namespace
using ttnn::operations::grid_sample::grid_sample;
using ttnn::operations::grid_sample::prepare_grid_sample_grid;

}  // namespace ttnn
