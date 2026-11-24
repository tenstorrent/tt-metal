// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"

/**
 * Convert linear flat index to zigzag flat index for per-head load balancing.
 *
 * In causal attention, Q chunks in the first half of each head process fewer KV
 * chunks than Q chunks in the second half. This creates work imbalance when
 * cores process consecutive Q chunks.
 *
 * Per-head zigzag interleaves light and heavy work:
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

/**
 * Result of decomposing a global Q index into (batch, head, q_chunk) coordinates.
 */
struct GlobalQIndex {
    uint32_t nb;
    uint32_t nq;
    uint32_t q_chunk;
};

/**
 * Decompose a linear global Q index in [0, B*NQH*num_q_chunks) into (nb, nq, q_chunk).
 *
 * Applies the remap (linear or zigzag) first, then splits the remapped index as:
 *   nb      = idx / (NQH * num_q_chunks)
 *   nq      = (idx / num_q_chunks) % NQH
 *   q_chunk = idx % num_q_chunks
 *
 * Used by SDPA reader/writer kernels under global Q scheduling to fetch the right
 * Q/K/V and write outputs for each (batch, head, q_chunk) triple a core is assigned.
 */
FORCE_INLINE GlobalQIndex decompose_global_q_index(uint32_t idx, uint32_t num_q_chunks, uint32_t NQH, bool use_zigzag) {
    const uint32_t remapped = remap_q_index(idx, num_q_chunks, use_zigzag);
    return {
        /*nb=*/remapped / (NQH * num_q_chunks),
        /*nq=*/(remapped / num_q_chunks) % NQH,
        /*q_chunk=*/remapped % num_q_chunks,
    };
}
