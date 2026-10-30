// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResScoresParams {
    uint32_t num_sites;
    float inv_hidden_size;
    float eps;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// stats - [1, C, N, 32 * P] float32, TILE layout: P tensor-parallel partials side by side,
//         as an all-gather on the last dim leaves them, or P = 1 already summed. Within a
//         partial, column 0 is a candidate's sum of squares and column 1 + r its dot with
//         query r.
struct AttnResScoresInputs {
    ttnn::Tensor stats;
};

}  // namespace ttnn::experimental::prim
