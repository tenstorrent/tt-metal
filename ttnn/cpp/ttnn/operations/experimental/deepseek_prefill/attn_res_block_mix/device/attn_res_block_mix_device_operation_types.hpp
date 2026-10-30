// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResBlockMixParams {
    uint32_t num_sites;
    float inv_hidden_size;
    float eps;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// sealed - [1, S_i, N, d] bfloat16 tensors, TILE, interleaved; their planes in order are the
//          C candidates.
// stats  - [1, C, N, 32 * P] float32, TILE: P tensor-parallel partials side by side, each
//          with the sum of squares in column 0 and query r's dot in column 1 + r.
struct AttnResBlockMixInputs {
    std::vector<ttnn::Tensor> sealed;
    ttnn::Tensor stats;
};

}  // namespace ttnn::experimental::prim
