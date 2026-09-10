// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResWeightedReduceNCParams {
    int32_t dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// input  - [1, C, H, W], TILE layout. The reduction runs over C.
// weight - [R, C, H, 1], TILE layout. One scalar per (r, c, h), physically a
//          tile whose column 0 holds it; that is already the layout
//          BroadcastType::COL reads, so no pre-pass builds it.
// output - [R, 1, H, W]. The weight's dim 0 is the output's batch: one reduction
//          of the same input against each of the R weight sets.
//
// Batching on the weight rather than the input is the point of the op. R sets
// against one input is the shape the caller has, and a kernel that knows the
// input is shared reads it once for a whole group of sets instead of once per
// set. An input-batched form would have nothing to reuse.
struct AttnResWeightedReduceNCInputs {
    ttnn::Tensor input;
    ttnn::Tensor weight;
};

}  // namespace ttnn::experimental::prim
