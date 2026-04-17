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
    // When true, distribute work as a single flat B*NQH*q_num_chunks range across cores
    // (matching ring_joint_sdpa's scheme). Default (false) keeps the hierarchical
    // batch -> heads -> q_chunks parallelization. Currently supported only for the
    // causal, non-chunked, no-attention-sink path.
    bool flatten_work = false;
};

}  // namespace ttnn::operations::transformer
