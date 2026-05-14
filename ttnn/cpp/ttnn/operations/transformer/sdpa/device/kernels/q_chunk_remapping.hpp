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
 * Ring-joint odd-Q schedule.
 *
 * For an odd per-head Q chunk count, one center chunk has no light/heavy pair.
 * This schedule gives each active core a parity-aligned paired region followed
 * by an optional unpaired center slot:
 *   [q0, qN-1], [q1, qN-2], ... [qB center]
 *
 * The paired region stays compatible with K multicast. The caller can
 * special-case the center slot when needed.
 */
FORCE_INLINE uint32_t ring_joint_linear_to_odd_global_zigzag(
    uint32_t linear_index,
    uint32_t num_q_chunks,
    uint32_t total_heads,
    uint32_t causal_q_chunk_boundary,
    uint32_t odd_schedule_pair_slots,
    uint32_t odd_schedule_centers_per_core) {
    const uint32_t schedule_stride = odd_schedule_pair_slots + odd_schedule_centers_per_core;
    const uint32_t core_idx = linear_index / schedule_stride;
    const uint32_t slot_idx = linear_index - core_idx * schedule_stride;
    const uint32_t pairs_per_core = odd_schedule_pair_slots / 2;

    if (slot_idx < odd_schedule_pair_slots) {
        const uint32_t pair_unit = core_idx * pairs_per_core + slot_idx / 2;
        if (pair_unit < total_heads * causal_q_chunk_boundary) {
            const uint32_t head_idx = pair_unit / causal_q_chunk_boundary;
            const uint32_t pair_idx = pair_unit - head_idx * causal_q_chunk_boundary;
            const uint32_t q_chunk = (slot_idx % 2 == 0) ? pair_idx : (num_q_chunks - 1 - pair_idx);
            return head_idx * num_q_chunks + q_chunk;
        }
        const uint32_t q_chunk = (slot_idx % 2 == 0) ? 0 : (num_q_chunks - 1);
        return q_chunk;
    }

    const uint32_t center_idx = core_idx * odd_schedule_centers_per_core + (slot_idx - odd_schedule_pair_slots);
    if (center_idx < total_heads) {
        return center_idx * num_q_chunks + causal_q_chunk_boundary;
    }

    return 0;
}

FORCE_INLINE bool ring_joint_is_odd_center_slot(
    uint32_t linear_index, uint32_t odd_schedule_pair_slots, uint32_t odd_schedule_centers_per_core) {
    const uint32_t schedule_stride = odd_schedule_pair_slots + odd_schedule_centers_per_core;
    const uint32_t slot_idx = linear_index % schedule_stride;
    return odd_schedule_centers_per_core != 0 && slot_idx >= odd_schedule_pair_slots;
}

FORCE_INLINE uint32_t remap_ring_joint_q_index(
    uint32_t linear_index,
    uint32_t num_q_chunks,
    uint32_t total_heads,
    uint32_t causal_q_chunk_boundary,
    bool use_zigzag,
    bool use_odd_global_q_schedule,
    uint32_t odd_schedule_pair_slots = 0,
    uint32_t odd_schedule_centers_per_core = 0) {
    if (use_odd_global_q_schedule) {
        return ring_joint_linear_to_odd_global_zigzag(
            linear_index,
            num_q_chunks,
            total_heads,
            causal_q_chunk_boundary,
            odd_schedule_pair_slots,
            odd_schedule_centers_per_core);
    }
    return remap_q_index(linear_index, num_q_chunks, use_zigzag);
}
