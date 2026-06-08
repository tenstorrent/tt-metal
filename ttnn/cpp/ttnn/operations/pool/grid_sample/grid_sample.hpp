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
 * Grid sample operation for spatial sampling with bilinear interpolation.
 *
 * Samples input tensor at grid locations using bilinear interpolation.
 * Grid coordinates are expected to be normalized to [-1, 1] range.
 *
 * Args:
 *   input_tensor: Input tensor of shape (N, C, H_in, W_in)
 *   grid: Sampling grid of shape (N, H_out, W_out, 2) with coordinates in [-1, 1]
 *   mode: Interpolation mode, currently only "bilinear" is supported
 *   padding_mode: How to handle out-of-bounds coordinates, currently only "zeros" is supported
 *   align_corners: Whether to align corners when mapping normalized coordinates to pixel indices
 *   use_precomputed_grid: Whether to use precomputed grid coordinates, currently only false is supported
 *   batch_output_channels: If true, fold output channels into the batch dimension
 *   memory_config: Memory configuration for the output tensor
 *   compute_kernel_config: Optional compute kernel configuration (math fidelity, fp32 dest
 *     accumulation, etc.). Defaults are arch-specific.
 *
 * Returns:
 *   Output tensor of shape (N, C, H_out, W_out)
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
