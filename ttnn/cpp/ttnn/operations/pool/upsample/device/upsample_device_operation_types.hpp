// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include <tuple>

namespace ttnn::prim {

struct UpsampleParams {
    float scale_factor_h = 1.0f;
    float scale_factor_w = 1.0f;
    std::string mode = "nearest";
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<ttnn::operations::sliding_window::SlidingWindowConfig> sliding_window_config = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "scale_factor_h",
        "scale_factor_w",
        "mode",
        "output_mem_config",
        "compute_kernel_config",
        "sliding_window_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            scale_factor_h, scale_factor_w, mode, output_mem_config, compute_kernel_config, sliding_window_config);
    }
};

}  // namespace ttnn::prim
