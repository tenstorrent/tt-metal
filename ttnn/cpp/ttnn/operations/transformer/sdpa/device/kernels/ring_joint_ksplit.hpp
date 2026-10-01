// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// K split for the chunked, non-sliding ring path: several cores share one (head, Q chunk) unit, each attends to a
// disjoint slice of every ring iteration's K chunks, and the last split (the reducer) merges their raw
// (max, sum, out) states and normalizes. The logical length is derived on device (trace replay), so a slice is a pure
// function of values reader, compute and writer all hold: ring id, logical tile count and split index.

#pragma once

#include <cstdint>

#include "dataflow/chunked_prefill_utils.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Bounds the reducer's serial merge, one pass per sender.
constexpr uint32_t kKSplitMaxCount = 4;

struct KSplitRange {
    uint32_t begin;
    uint32_t end;
    bool empty() const { return begin >= end; }
    bool contains(uint32_t k_chunk) const { return k_chunk >= begin && k_chunk < end; }
};

constexpr KSplitRange kKSplitAll{0, 0xFFFFFFFFu};

// Local K chunks of ring_id's shard that the K loop processes (chunked_kv_chunk_is_live; causal_end_nt is logical_nt
// without the causal skip). The local-to-global tile map is increasing, so they form a prefix.
template <
    bool kv_pad_rotation_enabled,
    bool chunked_enabled,
    uint32_t kv_local_padded_Nt,
    uint32_t chunk_size_t,
    uint32_t q_local_padded_Nt,
    uint32_t Sk_chunk_t>
inline uint32_t ksplit_valid_local_k_chunks(
    uint32_t num_local_k_chunks, uint32_t ring_id, uint32_t logical_nt, uint32_t causal_end_nt) {
    uint32_t lo = 0;
    uint32_t hi = num_local_k_chunks;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (chunked_kv_chunk_is_live<
                kv_pad_rotation_enabled,
                chunked_enabled,
                kv_local_padded_Nt,
                chunk_size_t,
                q_local_padded_Nt>(ring_id, mid, Sk_chunk_t, logical_nt, causal_end_nt)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Split i of n owns [i * valid / n, (i + 1) * valid / n). The last split always owns the final chunk, which holds the
// diagonal, so the reducer's rows are never fully masked.
inline KSplitRange ksplit_range(uint32_t num_valid, uint32_t split_idx, uint32_t split_count) {
    return {split_idx * num_valid / split_count, (split_idx + 1) * num_valid / split_count};
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
