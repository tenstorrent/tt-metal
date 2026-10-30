// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResBlockStatsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// sealed  - [1, S_i, N, d] bfloat16 tensors, TILE, interleaved; their planes in order are the
//           candidates.
// queries - [1, 1, d, 32] bfloat16, TILE: column 0 zero, query r in column 1 + r.
struct AttnResBlockStatsInputs {
    std::vector<ttnn::Tensor> sealed;
    ttnn::Tensor queries;
};

}  // namespace ttnn::experimental::prim
