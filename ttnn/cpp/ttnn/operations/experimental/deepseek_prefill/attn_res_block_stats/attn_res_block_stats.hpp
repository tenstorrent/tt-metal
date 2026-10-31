// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_stats {

// This rank's half of an AttnRes block's statistics, from one pass over the sealed set:
//
//   out[0][c][n][0]     = sum_d v[c][n][d]^2
//   out[0][c][n][1 + r] = sum_d v[c][n][d] * queries[d][1 + r]
//
// `sealed` is the sealed set as the snapshots were sealed: one or more `[1, S_i, N, d]`
// tensors whose planes, in order, are the C = sum S_i candidates. Taking them separately
// is what lets a seal append a snapshot instead of copying the set to grow it.
//
// `queries` is `[1, 1, d, 32]` with column 0 zero and query r in column 1 + r, so one
// accumulator takes both reductions: the dots land in the query columns and the sum of
// squares, a matmul of the squared row against a column of ones, in column 0.
//
// With `cluster_axis`, the statistics are gathered across that mesh axis in the same
// program: each rank's land in tile column `rank` of `gathered`, `[1, C, N, 32 * P]`, which is
// returned, and a rank's rows go to its peers as they are summed rather than after the
// pass. `gathered` must sit at one address on every chip and stay allocated across calls:
// peers write into it by address, and a buffer freed and reallocated between two calls can
// still be in use on a chip that has not caught up.
ttnn::Tensor attn_res_block_stats(
    const std::vector<ttnn::Tensor>& sealed,
    const ttnn::Tensor& queries,
    const std::optional<ttnn::Tensor>& gathered,
    std::optional<uint32_t> cluster_axis,
    std::optional<ttnn::ccl::Topology> topology,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::operations::experimental::deepseek_prefill::attn_res_block_stats
