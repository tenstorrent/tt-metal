// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "grid_sample_prepare_grid.hpp"

namespace ttnn {

namespace operations::grid_sample {

ttnn::Tensor grid_sample(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& grid,
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool align_corners = false,
    bool use_precomputed_grid = false,
    bool batch_output_channels = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace operations::grid_sample

// Import functions to main ttnn namespace
using ttnn::operations::grid_sample::grid_sample;
using ttnn::operations::grid_sample::prepare_grid_sample_grid;

}  // namespace ttnn
