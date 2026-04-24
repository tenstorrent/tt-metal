// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::transformer {

// Single-chip perf proxy for multi-chip ring-joint SDPA. All non-None cases distribute work flat
// across cores (B * NQH * q_num_chunks split evenly) instead of hierarchical batch × heads × q,
// matching what ring_joint_sdpa does per device.
//
// - None: hierarchical batch/heads/q split (default).
// - Diag: flat; full Q × full K. Mirrors the iter-0 (diag) work on each ring device. Requires
//         is_causal=true.
// - Up:   flat; full Q, K-loop capped at k_num_chunks/2. Mirrors a non-diag iter where
//         ring_index > ring_id (half-K per Q). Requires is_causal=false.
// - Down: flat; only the heavy Q half (q_num_chunks/2 slots), full K per slot. Mirrors a non-diag
//         iter where ring_index < ring_id (is_balanced Q-skip). Requires is_causal=false.
enum class RingProxyCase : uint8_t {
    None = 0,
    Diag = 1,
    Up = 2,
    Down = 3,
};

struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::optional<CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    RingProxyCase ring_proxy_case = RingProxyCase::None;
};

}  // namespace ttnn::operations::transformer
