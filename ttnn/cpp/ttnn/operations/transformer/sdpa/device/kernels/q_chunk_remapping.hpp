// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"

/**
 * Convert linear flat index to zigzag flat index for per-head load balancing.
 *
 * In causal attention with is_balanced=true, Q chunks in the first half of each head
 * (positions 0 to num_q_chunks/2-1) process fewer KV chunks than Q chunks in the
 * second half. This creates work imbalance when cores process consecutive Q chunks.
 *
 * Per-head zigzag interleaves light and heavy work within each core:
 *   - Even positions (0, 2, 4, ...) map to forward indices: 0, 1, 2, 3, ...
 *   - Odd positions (1, 3, 5, ...) map to backward indices: N-1, N-2, N-3, ...
 *
 * Example with num_q_chunks=20:
 *   pos 0 -> q_chunk 0  (light)
 *   pos 1 -> q_chunk 19 (heavy)
 *   pos 2 -> q_chunk 1  (light)
 *   pos 3 -> q_chunk 18 (heavy)
 *   ...
 *
 * This works for ANY division of work across cores (no exact division constraint).
 */
FORCE_INLINE uint32_t linear_to_zigzag(uint32_t linear_flat, uint32_t num_q_chunks) {
    const uint32_t head_idx = linear_flat / num_q_chunks;
    const uint32_t pos_in_head = linear_flat % num_q_chunks;

    uint32_t q_chunk;
    if (pos_in_head % 2 == 0) {
        // Even positions: forward from start
        q_chunk = pos_in_head / 2;
    } else {
        // Odd positions: backward from end
        q_chunk = num_q_chunks - 1 - (pos_in_head / 2);
    }

    return head_idx * num_q_chunks + q_chunk;
}

/**
 * Remap Q index for work balancing.
 *
 * When use_zigzag is true, applies zigzag remapping to balance light/heavy work.
 * When false, returns the linear index unchanged.
 *
 * With -O3, the compiler eliminates the dead branch when use_zigzag is constexpr.
 */
FORCE_INLINE uint32_t remap_q_index(uint32_t linear_index, uint32_t num_q_chunks, bool use_zigzag) {
    if (use_zigzag) {
        return linear_to_zigzag(linear_index, num_q_chunks);
    }
    return linear_index;
}

// Single-chip ring-iter proxy. Values mirror ttnn::operations::transformer::RingProxyCase and
// are passed from the host as a compile-time arg. None => hierarchical (no flat work).
enum class RingProxyMode : uint8_t { None = 0, Diag = 1, Up = 2, Down = 3 };

FORCE_INLINE constexpr bool proxy_uses_flat_work(RingProxyMode m) { return m != RingProxyMode::None; }

struct FlatQIndex {
    uint32_t nb;
    uint32_t nq;
    uint32_t q_chunk;
};

// Effective Q chunk count + offset for a given proxy mode. Down halves the Q space and points at
// the heavy half; Diag / Up / None keep the full range at offset 0.
struct ProxyQRange {
    uint32_t q_num_effective;
    uint32_t q_chunk_offset;
};

FORCE_INLINE constexpr ProxyQRange proxy_q_range(uint32_t q_num_chunks, RingProxyMode proxy) {
    if (proxy == RingProxyMode::Down) {
        return {q_num_chunks / 2, q_num_chunks / 2};
    }
    return {q_num_chunks, 0};
}

// Decompose a flat index in [0, B*NQH*q_num_effective) into (nb, nq, q_chunk). q_chunk is shifted
// into the physical Q range (so Down returns the heavy half directly).
FORCE_INLINE FlatQIndex decompose_flat_q_index(
    uint32_t linear_index, uint32_t q_num_chunks, uint32_t NQH, bool use_zigzag, RingProxyMode proxy) {
    const ProxyQRange range = proxy_q_range(q_num_chunks, proxy);
    const uint32_t flat = remap_q_index(linear_index, range.q_num_effective, use_zigzag);
    return {
        /*nb=*/flat / (NQH * range.q_num_effective),
        /*nq=*/(flat / range.q_num_effective) % NQH,
        /*q_chunk=*/flat % range.q_num_effective + range.q_chunk_offset,
    };
}

// Compute-side q_chunk lookup. Compute doesn't need the (nb, nq) split — reader/writer do.
FORCE_INLINE uint32_t proxy_q_chunk(uint32_t q_iter, uint32_t q_num_chunks, bool use_zigzag, RingProxyMode proxy) {
    const ProxyQRange range = proxy_q_range(q_num_chunks, proxy);
    return remap_q_index(q_iter, range.q_num_effective, use_zigzag) % range.q_num_effective + range.q_chunk_offset;
}

// Hierarchical iteration: decompose a linear gq ∈ [0, B_local * H_local * Q_per_core) into
// (nb_idx, nq_idx, q_iter) in lexicographic order (nb outer, nq middle, q_iter inner).
struct HierarchicalIndex {
    uint32_t nb_idx;
    uint32_t nq_idx;
    uint32_t q_iter;
};
FORCE_INLINE HierarchicalIndex
decompose_hierarchical_index(uint32_t gq, uint32_t heads_local, uint32_t q_chunks_per_core) {
    return {
        /*nb_idx=*/gq / (q_chunks_per_core * heads_local),
        /*nq_idx=*/(gq / q_chunks_per_core) % heads_local,
        /*q_iter=*/gq % q_chunks_per_core,
    };
}

// Pick the q_chunk for a given q_iter slot. BALANCED_Q_PARALLEL (causal + even chunks) interleaves
// light and heavy halves so cores see even work; plain mode is consecutive.
FORCE_INLINE uint32_t balanced_q_chunk(
    uint32_t q_iter, uint32_t local_q_start, uint32_t q_chunks_per_core, uint32_t q_num_chunks, bool balanced) {
    if (!balanced) {
        return local_q_start + q_iter;
    }
    const uint32_t half = q_chunks_per_core / 2;
    if (q_iter < half) {
        return local_q_start + q_iter;  // bottom half: forward from local_q_start
    }
    return q_num_chunks - 1 - (local_q_start + (q_iter - half));  // top half: backward from end
}
