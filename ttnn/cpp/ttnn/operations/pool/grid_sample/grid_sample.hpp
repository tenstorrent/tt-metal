// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "grid_sample_prepare_grid.hpp"

namespace ttnn {
namespace operations {
namespace grid_sample {

struct ExecuteGridSample {
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
     *   use_precomputed_grid: Whether to use precomputed grid coordinates, currently only false is supported
     *   memory_config: Memory configuration for the output tensor
     *
     * Returns:
     *   Output tensor of shape (N, C, H_out, W_out)
     */
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& grid,
        const std::string& mode = "bilinear",
        const std::string& padding_mode = "zeros",
        bool use_precomputed_grid = false,
        bool batch_output_channels = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace grid_sample
}  // namespace operations

// Register the operation
constexpr auto grid_sample =
    ttnn::register_operation<"ttnn::grid_sample", ttnn::operations::grid_sample::ExecuteGridSample>();

// Import prepare function to main ttnn namespace
using ttnn::operations::grid_sample::prepare_grid_sample_grid;

}  // namespace ttnn
