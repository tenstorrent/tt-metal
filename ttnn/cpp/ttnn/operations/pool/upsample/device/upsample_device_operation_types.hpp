// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::prim {

struct UpsampleParams {
    float scale_factor_h = 1.0f;
    float scale_factor_w = 1.0f;
    std::string mode = "nearest";
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<ttnn::operations::sliding_window::SlidingWindowConfig> sliding_window_config = std::nullopt;
};

}  // namespace ttnn::prim
