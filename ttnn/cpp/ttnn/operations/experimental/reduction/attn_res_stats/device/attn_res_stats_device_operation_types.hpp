// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResStatsParams {
    tt::tt_metal::DataType dtype;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// v - [1, C, N, D], TILE layout. The candidate keys, sharded on `d`.
// q - [1, 1, 1, D], TILE layout. One query row, broadcast down the tokens.
struct AttnResStatsInputs {
    ttnn::Tensor v;
    ttnn::Tensor q;
};

}  // namespace ttnn::experimental::prim
