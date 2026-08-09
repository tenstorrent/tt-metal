// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResMergeParams {
    uint32_t site;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// partial - [R, 1, H, W], TILE layout. prefix_sum - [1, 1, H, W]: the live
//   stream is one plane behind every site, so only the partial batches.
// shift, mass, live_scores - [R, 1, H, 1], TILE layout. One scalar per row,
//   physically a tile whose column 0 holds it; that is already the layout
//   BroadcastType::COL reads, so no pre-pass builds it.
//
// An operand's dim 0 is a read-site axis, and `site` picks the plane. At R == 1
// the operand is shared by every site and `site` does not apply to it, which is
// what lets one call mix a batched partial, shift and mass with a live_scores
// that was computed for this site alone. The output is one plane either way.
struct AttnResMergeInputs {
    ttnn::Tensor partial;
    ttnn::Tensor prefix_sum;
    ttnn::Tensor shift;
    ttnn::Tensor mass;
    ttnn::Tensor live_scores;
};

}  // namespace ttnn::experimental::prim
