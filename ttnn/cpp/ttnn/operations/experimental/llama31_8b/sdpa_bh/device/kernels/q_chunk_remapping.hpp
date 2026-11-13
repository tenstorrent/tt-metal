// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"

FORCE_INLINE uint32_t linear_to_zigzag(uint32_t linear_flat, uint32_t num_q_chunks) {
    const uint32_t head_idx = linear_flat / num_q_chunks;
    const uint32_t pos_in_head = linear_flat % num_q_chunks;

    uint32_t q_chunk;
    if (pos_in_head % 2 == 0) {
        q_chunk = pos_in_head / 2;
    } else {
        q_chunk = num_q_chunks - 1 - (pos_in_head / 2);
    }

    return head_idx * num_q_chunks + q_chunk;
}

inline uint32_t q_unit_table[Q_UNIT_TABLE_SIZE];

FORCE_INLINE uint32_t remap_q_index(uint32_t linear_index, uint32_t, bool) { return q_unit_table[linear_index]; }

struct GlobalQIndex {
    uint32_t nb;
    uint32_t nq;
    uint32_t q_chunk;
};

FORCE_INLINE GlobalQIndex decompose_global_q_index(uint32_t idx, uint32_t num_q_chunks, uint32_t NQH, bool use_zigzag) {
    const uint32_t remapped = remap_q_index(idx, num_q_chunks, use_zigzag);
    return {
        remapped / (NQH * num_q_chunks),
        (remapped / num_q_chunks) % NQH,
        remapped % num_q_chunks,
    };
}
