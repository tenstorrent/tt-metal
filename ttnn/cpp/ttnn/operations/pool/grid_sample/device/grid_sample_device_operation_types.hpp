// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::pool::grid_sample {

struct GridSampleParams {
    std::string mode = "bilinear";
    std::string padding_mode = "zeros";
    bool align_corners = false;
    bool use_precomputed_grid = false;
    bool batch_output_channels = false;
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "mode", "padding_mode", "align_corners", "use_precomputed_grid", "batch_output_channels", "output_mem_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->mode,
            this->padding_mode,
            this->align_corners,
            this->use_precomputed_grid,
            this->batch_output_channels,
            this->output_mem_config);
    }
};

struct GridSampleInputs {
    Tensor input_tensor;
    Tensor grid;
};

}  // namespace ttnn::operations::pool::grid_sample
