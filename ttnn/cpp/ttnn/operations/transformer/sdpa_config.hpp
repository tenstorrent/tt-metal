// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::transformer {

struct SDPAProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::optional<CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    // Ring joint SDPA only: split every (head, Q chunk) over this many key partitions (K chunk k belongs to
    // partition k % ring_k_split), each on its own core, so the work divides evenly over the grid. With > 1 the op
    // returns per-partition raw state instead of the normalized output: output [B, NH * ring_k_split, S, DV] holds
    // the unnormalized O of partition p at virtual head p * NH + h, stats the matching running max / sum; the caller
    // merges them (see ttnn.transformer.ring_joint_sdpa_merge_k_split).
    uint32_t ring_k_split = 1;
};

// Paired geometry for an HMA-shared paged K/V cache (chunked prefill SDPA and
// paged decode SDPA). When the physical buffer was allocated for a different layer's
// (num_kv_heads, block_size, head_dim) view — e.g. vLLM hybrid kv-cache-groups — the
// reader must address it with this call's block_size / num_kv_heads (Q drives head_dim).
// Presence of the override is the outer std::optional on the op API; both fields are
// required plain values whenever the struct is provided. Default {0, 0} means inactive
// after the optional is collapsed into operation attributes.
struct PagedCacheGeometryOverride {
    uint32_t block_size = 0;
    uint32_t num_kv_heads = 0;

    [[nodiscard]] bool active() const { return block_size != 0 || num_kv_heads != 0; }
};

}  // namespace ttnn::operations::transformer
