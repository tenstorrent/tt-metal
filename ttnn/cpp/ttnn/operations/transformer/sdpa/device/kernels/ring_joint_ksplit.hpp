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

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Bounds the reducer's serial merge, one pass per sender.
constexpr uint32_t kKSplitMaxCount = 4;

// [begin, end) of K chunks.
struct KSplitRange {
    uint32_t begin;
    uint32_t end;
    bool empty() const { return begin >= end; }
    bool contains(uint32_t i) const { return i >= begin && i < end; }
};

constexpr KSplitRange kKSplitAll{0, 0xFFFFFFFFu};

// Local K chunks of ring_id's shard that the K loop processes (chunked_kv_chunk_is_live; causal_end_nt is logical_nt
// without the causal skip). kv_region_Nt is a source's block-cyclic region (the Q slab over the stripe split). The
// local-to-global tile map is increasing, so they form a prefix.
template <
    bool kv_pad_rotation_enabled,
    bool chunked_enabled,
    uint32_t kv_local_padded_Nt,
    uint32_t chunk_size_t,
    uint32_t kv_region_Nt,
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
                kv_region_Nt>(ring_id, mid, Sk_chunk_t, logical_nt, causal_end_nt)) {
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

// First chunk of `band` in a split of `num_chunks` chunks into `bands`, for the slice of tensor rank `key`. Every band
// gets num_chunks / bands chunks, and the num_chunks % bands extra ones go to the bands from key % bands on (wrapping),
// so over the ring's slices the remainders rotate across bands. Fewer chunks than bands keep the floor split
// [i * n / bands, (i + 1) * n / bands), so the last band still holds the last chunk.
inline uint32_t band_begin(uint32_t num_chunks, uint32_t bands, uint32_t band, uint32_t key) {
    if (num_chunks < bands) {
        return band * num_chunks / bands;
    }
    const uint32_t extra = num_chunks % bands;
    const uint32_t first_extra = key % bands;
    // Extra bands are first_extra .. first_extra + extra - 1 (mod bands); count those below `band`.
    uint32_t extras_before = 0;
    for (uint32_t k = 0; k < extra; ++k) {
        extras_before += (first_extra + k) % bands < band ? 1 : 0;
    }
    return band * (num_chunks / bands) + extras_before;
}

// Band split_idx of the num_valid chunks of ring_id's slice, its remainder chunks rotated over the ring (band_begin).
// The last band ends at num_valid, so it always holds the last valid chunk (the diagonal) and the reducer has state.
inline KSplitRange ksplit_range(uint32_t num_valid, uint32_t split_idx, uint32_t split_count, uint32_t ring_id) {
    return {
        band_begin(num_valid, split_count, split_idx, ring_id),
        band_begin(num_valid, split_count, split_idx + 1, ring_id)};
}

// Sliding split of a (head, Q chunk) unit's work plan. A plan shorter than two chunks per band (the first window of the
// sequence) stays whole on the reducer, the band that holds the diagonal chunk.
inline KSplitRange sliding_ksplit_range(uint32_t num_items, uint32_t split_idx, uint32_t split_count) {
    if (num_items < 2 * split_count) {
        return split_idx + 1 == split_count ? KSplitRange{0, num_items} : KSplitRange{0, 0};
    }
    return ksplit_range(num_items, split_idx, split_count);
}

// Slot donation (ring_joint_donation.hpp): every home core gives one ring iteration of its band (a slot) to a spare
// core. The donation runtime args, the same block in the reader and compute: [the donated slot's steps (a home's; 0 if
// none), visit count | total visit steps << 8 (a spare's; 0 if none), then per visit one word]. Steps are ring
// iterations, packed as first step | step count << 8; a visit as its steps | unit << 16 | band << 24. The writer's
// block holds [the physical x | y << 16 of the spare visiting this core (0 if none), visit count, then per visit its
// home's]. A spare signals kDonationOrdinal on its home's K-split semaphore once a visit is staged in its half A; the
// home pulls it into its restore CBs and bumps the spare's donation_pulled_sem, which frees half A.
constexpr uint32_t kDonationMaxVisits = 8;
constexpr uint32_t kDonationArgWords = 2 + kDonationMaxVisits;
constexpr uint32_t kDonationWriterArgWords = 2 + kDonationMaxVisits;
// The K-split semaphore ordinal a spare signals (senders use 1 .. kKSplitMaxCount - 1).
constexpr uint32_t kDonationOrdinal = kKSplitMaxCount;
// Ring iterations whose IDs a core keeps (a spare visits slots out of order); the host caps ring_size at this.
constexpr uint32_t kMaxRingIterations = 32;

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
