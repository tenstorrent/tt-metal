// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::transformer {

struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::optional<CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    // Flat work distribution: treat (batch, head, q_chunk) as one linear space and split it evenly
    // across cores. Use for workloads where the hierarchical batch -> heads -> q_chunks split leaves
    // cores idle (e.g. low batch × head product). Default (false) keeps the hierarchical
    // parallelization. Currently supported only for the causal, non-chunked, no-attention-sink path.
    //
    // Note: at ring iter 0 of a causal + balanced ring SDPA, each device runs plain causal SDPA on
    // its local Q/K/V with this same flat distribution, so flatten_work=true makes a single-chip
    // SDPA an equivalent perf proxy for that iteration — useful for measuring per-device work
    // without a multi-chip setup.
    bool flatten_work = false;
};

}  // namespace ttnn::operations::transformer
