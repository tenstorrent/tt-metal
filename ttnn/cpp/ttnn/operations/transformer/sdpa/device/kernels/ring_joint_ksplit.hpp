// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// K split for the chunked ring path: several cores share one (head, Q chunk) unit, each attends to a disjoint slice of
// every ring iteration's K chunks (sliding: a band of the unit's work plan), and the last split (the reducer) merges
// their raw (max, sum, out) states and normalizes. The logical length is derived on device (trace replay), so a slice
// is a pure function of values reader, compute and writer all hold: ring id, logical tile count and split index.

#pragma once

#include <cstdint>

#include "dataflow/chunked_prefill_utils.hpp"
#include "../../../../experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_slice_readiness.hpp"

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

// Band split_idx of a slice of tensor rank `ring_id`: the gather's band split (band_begin), so the K split's remainder
// chunks rotate over the bands across the ring and each band's chunks arrive in the readiness parts it reads them in.
inline KSplitRange ksplit_range(uint32_t num_valid, uint32_t split_idx, uint32_t split_count, uint32_t ring_id) {
    return {
        ttnn::ring_attention_all_gather::band_begin(num_valid, split_count, split_idx, ring_id),
        ttnn::ring_attention_all_gather::band_begin(num_valid, split_count, split_idx + 1, ring_id)};
}

// Sliding split of a (head, Q chunk) unit's work plan. A plan shorter than two chunks per band (the first window of the
// sequence) stays whole on the reducer, the band that holds the diagonal chunk.
inline KSplitRange sliding_ksplit_range(uint32_t num_items, uint32_t split_idx, uint32_t split_count) {
    if (num_items < 2 * split_count) {
        return split_idx + 1 == split_count ? KSplitRange{0, num_items} : KSplitRange{0, 0};
    }
    return ksplit_range(num_items, split_idx, split_count);
}

// Interleaved ring steps (ring_interleave): a remote slice lands at one link's rate, while one ring iteration would use
// it faster, so each pair of consecutive remote iterations (1, 2), (3, 4), ... (one slice per direction) runs as
// kInterleavePieces steps of each, alternating, every step one piece of the core's K-split slice. Both directions'
// slices are then used at half rate. Iteration 0 (local) and an unpaired last iteration run whole. Reader and compute
// walk the same steps; the writer, which has no per-iteration work for single-Q K-split cores, keeps ring iterations.
constexpr uint32_t kInterleavePieces = 2;

struct InterleavedStep {
    uint32_t ring_iter;
    uint32_t piece;
    uint32_t pieces;
};

inline uint32_t interleaved_step_count(uint32_t ring_iterations) {
    const uint32_t remote = ring_iterations - 1;
    return 1 + (remote / 2) * 2 * kInterleavePieces + remote % 2;
}

inline InterleavedStep interleaved_step(uint32_t step, uint32_t ring_iterations) {
    if (step == 0) {
        return {0, 0, 1};
    }
    const uint32_t paired_steps = ((ring_iterations - 1) / 2) * 2 * kInterleavePieces;
    const uint32_t s = step - 1;
    if (s < paired_steps) {
        const uint32_t r = s % (2 * kInterleavePieces);
        return {1 + 2 * (s / (2 * kInterleavePieces)) + r % 2, r / 2, kInterleavePieces};
    }
    return {ring_iterations - 1, 0, 1};
}

// Piece `piece` of `pieces` of a K-split slice (the readiness parts land in this order, see
// ring_attention_slice_readiness.hpp).
inline KSplitRange interleaved_piece(KSplitRange range, uint32_t piece, uint32_t pieces) {
    const uint32_t len = range.end > range.begin ? range.end - range.begin : 0;
    return {range.begin + len * piece / pieces, range.begin + len * (piece + 1) / pieces};
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
