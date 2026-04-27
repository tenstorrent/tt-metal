// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared between host (SDPAProgramConfig) and kernel (q_chunk_remapping.hpp). Keep this header
// dependency-free so kernel translation units can include it.

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

}  // namespace ttnn::operations::transformer
