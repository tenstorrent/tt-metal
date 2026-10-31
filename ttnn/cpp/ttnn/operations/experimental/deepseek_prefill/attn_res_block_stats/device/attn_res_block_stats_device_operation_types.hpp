// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResBlockStatsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    // With an axis, the op gathers the statistics across it itself: every rank's land side
    // by side in `gathered`, the layout `attn_res_block_mix` reads. The axis's topology
    // decides how far a multicast reaches each way, and the gather stripes across `num_links`.
    std::optional<uint32_t> cluster_axis;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear;
    uint32_t num_links = 1;
};

// sealed   - [1, S_i, N, d] bfloat16 tensors, TILE, interleaved; their planes in order are the
//            candidates.
// queries  - [1, 1, d, 32] bfloat16, TILE: column 0 zero, query r in column 1 + r.
// gathered - with `cluster_axis`, [1, C, N, 32 * P] float32, TILE, interleaved: rank p's
//            statistics in tile column p. Written in place, by this rank and by its peers.
struct AttnResBlockStatsInputs {
    std::vector<ttnn::Tensor> sealed;
    ttnn::Tensor queries;
    std::optional<ttnn::Tensor> gathered;
};

}  // namespace ttnn::experimental::prim
