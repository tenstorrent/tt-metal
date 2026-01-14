// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::pool::upsample {

struct operation_attributes_t {
    int scale_factor_h = 0;
    int scale_factor_w = 0;
    std::string mode = "nearest";
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<sliding_window::SlidingWindowConfig> sliding_window_config = std::nullopt;
};

struct tensor_args_t {
   Tensor input_tensor;
};

}  // namespace ttnn::operations::pool::upsample
