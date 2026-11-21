// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The per-user work plan of paged_scaled_dot_product_attention_decode, shared by the reader, the compute and the
// writer kernel (single source: every kernel derives the same split from the same cur_pos value).
//
// For one user the key/value heads' position tiles form one virtual key sequence of
//     N = kv_heads * user_pos_tiles   units,   unit v = (head v / user_pos_tiles, position tile v % user_pos_tiles),
// with user_pos_tiles = cur_pos / 32 + 1 (cur_pos clamped into the cache: rule R6 leaves invalid values
// unspecified but forbids reading outside the caches). The sequence is split over the workers of the user's
// cluster group: G_active = clusters_for(N) clusters of C NEOs; worker w = rank * C + neo gets a contiguous,
// balanced range. A cluster with rank >= G_active skips the user (no ring traffic on any kernel); a worker with
// no units emits a neutral partial. The host mirrors clusters_for to size the plan for the largest position.

#pragma once

#include <cstdint>

namespace decode_plan {

constexpr uint32_t TILE_DIM = 32;
// Finite masks (never -inf: an all-masked row would give exp(-inf + inf) = NaN).
// NEG_BIG: added to a score that must not be attended (-1e9). NEG_EMPTY: the row max of an empty partial (-1e30).
constexpr uint32_t NEG_BIG_FP32_BITS = 0xCE6E6B28u;
constexpr uint32_t NEG_EMPTY_FP32_BITS = 0xF149F2CAu;

inline uint32_t isqrt_ceil(uint32_t x) {
    uint32_t r = 0;
    while (r * r < x) {
        ++r;
    }
    return r;
}

// Clusters that minimize N/W + kappa*W (phase 1 + phase 2) for N units, W = clusters * C workers, capped.
inline uint32_t clusters_for(uint32_t units, uint32_t compute_threads, uint32_t cluster_cap, uint32_t merge_cost) {
    const uint32_t workers = isqrt_ceil((units + merge_cost - 1) / merge_cost);
    uint32_t clusters = (workers + compute_threads - 1) / compute_threads;
    if (clusters < 1) {
        clusters = 1;
    }
    return clusters > cluster_cap ? cluster_cap : clusters;
}

struct UserPlan {
    uint32_t cur_pos;         // clamped into [0, tensor_pos_tiles * 32 - 1]
    uint32_t user_pos_tiles;  // position tiles attended: cur_pos / 32 + 1
    uint32_t last_col;        // cur_pos % 32: columns after it in the last tile are masked
    uint32_t units;           // N
    uint32_t workers_active;  // W_active = G_active * C
    bool cluster_active;      // rank < G_active
    uint32_t unit_start;      // this worker's range
    uint32_t unit_count;
};

template <
    uint32_t tensor_pos_tiles,
    uint32_t kv_heads,
    uint32_t compute_threads,
    uint32_t clusters_per_user,
    uint32_t merge_cost>
inline UserPlan make_plan(uint32_t raw_pos_bits, uint32_t cluster_rank, uint32_t neo) {
    UserPlan p{};
    const int32_t raw = static_cast<int32_t>(raw_pos_bits);
    constexpr uint32_t max_pos = tensor_pos_tiles * TILE_DIM - 1;
    p.cur_pos = raw < 0 ? 0u : (static_cast<uint32_t>(raw) > max_pos ? max_pos : static_cast<uint32_t>(raw));
    p.user_pos_tiles = p.cur_pos / TILE_DIM + 1;
    p.last_col = p.cur_pos % TILE_DIM;
    p.units = kv_heads * p.user_pos_tiles;
    const uint32_t clusters_active = clusters_for(p.units, compute_threads, clusters_per_user, merge_cost);
    p.workers_active = clusters_active * compute_threads;
    p.cluster_active = cluster_rank < clusters_active;
    const uint32_t w = cluster_rank * compute_threads + neo;
    const uint32_t base = p.units / p.workers_active;
    const uint32_t rem = p.units % p.workers_active;
    p.unit_count = p.cluster_active ? base + (w < rem ? 1u : 0u) : 0u;
    p.unit_start = w * base + (w < rem ? w : rem);
    return p;
}

// Walk a worker's range as head segments (a run inside one head) cut into chunks of at most k_chunk_tiles
// position tiles. on_segment(head, first_tile, len) / on_chunk(head, first_tile, n, ends_head) / on_segment_end().
// The tile that holds cur_pos (t = user_pos_tiles - 1) is always the last tile of its chunk.
template <uint32_t k_chunk_tiles, typename SegFn, typename ChunkFn, typename SegEndFn>
inline void for_each_chunk(const UserPlan& p, SegFn&& on_segment, ChunkFn&& on_chunk, SegEndFn&& on_segment_end) {
    const uint32_t upt = p.user_pos_tiles;
    uint32_t v = p.unit_start;
    const uint32_t end = p.unit_start + p.unit_count;
    while (v < end) {
        const uint32_t head = v / upt;
        const uint32_t first_tile = v % upt;
        const uint32_t head_end = (head + 1) * upt;
        const uint32_t seg_end = end < head_end ? end : head_end;
        const uint32_t len = seg_end - v;
        on_segment(head, first_tile, len);
        for (uint32_t off = 0; off < len; off += k_chunk_tiles) {
            const uint32_t n = (len - off) < k_chunk_tiles ? (len - off) : k_chunk_tiles;
            const uint32_t t0 = first_tile + off;
            on_chunk(head, t0, n, t0 + n == upt);
        }
        on_segment_end();
        v = seg_end;
    }
}

}  // namespace decode_plan
