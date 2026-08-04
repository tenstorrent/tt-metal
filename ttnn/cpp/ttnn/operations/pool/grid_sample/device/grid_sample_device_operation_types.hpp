// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct GridSampleParams {
    std::string mode = "bilinear";
    std::string padding_mode = "zeros";
    bool align_corners = false;
    bool use_precomputed_grid = false;
    bool batch_output_channels = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config{};

    static constexpr auto attribute_names = std::forward_as_tuple(
        "mode",
        "padding_mode",
        "align_corners",
        "use_precomputed_grid",
        "batch_output_channels",
        "output_mem_config",
        "compute_kernel_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->mode,
            this->padding_mode,
            this->align_corners,
            this->use_precomputed_grid,
            this->batch_output_channels,
            this->output_mem_config,
            this->compute_kernel_config);
    }
};

struct GridSampleInputs {
    Tensor input_tensor;
    Tensor grid;
};

}  // namespace ttnn::prim
