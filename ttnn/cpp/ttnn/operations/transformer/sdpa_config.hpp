// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::transformer {

// Single-chip perf proxy for multi-chip ring-joint SDPA iter 1+ work. The proxy keeps full-size
// Q/K/V (matching what each device holds at iter 0) and skips reads/compute on the same pattern
// the ring kernel uses for non-diag iters, so DRAM layout, tile addresses, and core dispatch match
// the ring per-iter work exactly.
//
// - None: disabled (default). Behavior matches prior releases.
// - Up:   simulate ring iter where ring_index > ring_id (skips upper half of K per Q). K-loop
//         caps at k_num_chunks / 2; every Q chunk is still assigned.
// - Down: simulate ring iter where ring_index < ring_id (skips light Q stripe). Only the heavy
//         half of Q chunks is assigned; K-loop stays full-length.
//
// Requires flatten_work=true, is_causal=false, and keeps tensor shapes symmetric with iter-0
// (full L×L).
enum class RingProxyCase : uint8_t {
    None = 0,
    Up = 1,
    Down = 2,
};

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
    RingProxyCase ring_proxy_case = RingProxyCase::None;
};

}  // namespace ttnn::operations::transformer
