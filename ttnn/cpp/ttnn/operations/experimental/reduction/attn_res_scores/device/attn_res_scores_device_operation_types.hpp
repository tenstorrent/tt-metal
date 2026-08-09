// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResScoresParams {
    float inv_hidden_size;
    float eps;
    tt::tt_metal::DataType dtype;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// stats - [1, 2C, N, W], TILE layout. Sums of squares in candidates [0, C),
//   dots in [C, 2C); they arrive stacked because one collective reduces both.
struct AttnResScoresInputs {
    ttnn::Tensor stats;
};

}  // namespace ttnn::experimental::prim
