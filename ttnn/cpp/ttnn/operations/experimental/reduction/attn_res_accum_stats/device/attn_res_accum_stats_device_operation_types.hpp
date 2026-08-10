// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResAccumStatsParams {
    tt::tt_metal::DataType stats_dtype;
    tt::tt_metal::MemoryConfig total_mem_config;
    tt::tt_metal::MemoryConfig stats_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// a, b - [1, C, N, D], TILE layout. The two halves of the residual sum, sharded on `d`.
// q     - [1, 1, 1, D], TILE layout. One query row, broadcast down the tokens.
struct AttnResAccumStatsInputs {
    ttnn::Tensor a;
    ttnn::Tensor b;
    ttnn::Tensor q;
};

}  // namespace ttnn::experimental::prim
