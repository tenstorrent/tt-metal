// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct FastWeightedReduceNCParams {
    int32_t dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// input  - [B, C, H, W], TILE layout. The reduction runs over C.
// weight - [B, C, H, 1], TILE layout. One scalar per (b, c, h), physically a
//          tile whose column 0 holds it; that is already the layout
//          BroadcastType::COL reads, so no pre-pass builds it.
struct FastWeightedReduceNCInputs {
    ttnn::Tensor input;
    ttnn::Tensor weight;
};

}  // namespace ttnn::experimental::prim
