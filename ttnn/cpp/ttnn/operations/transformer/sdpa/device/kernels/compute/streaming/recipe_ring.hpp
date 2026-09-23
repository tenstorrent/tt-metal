// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "recipe_checkpoint.hpp"
#include "../../dataflow/chunked_prefill_utils.hpp"

template <uint32_t scale, uint32_t subblock_h>
void sdpa_recipe_ring_segment(
    RecipeAccumulatorState& resident,
    uint32_t q_begin,
    uint32_t q_end,
    uint32_t local_chunks,
    uint32_t total_chunks,
    uint32_t primary_rows,
    uint32_t joint_rows,
    bool first_ring,
    bool last_ring) {
    const bool staged = q_end - q_begin > 1;
    const uint32_t valid_chunks = (primary_rows + 511) / 512 + (joint_rows + 511) / 512;
    ASSERT(valid_chunks > 0);
    for (uint32_t q = q_begin; q < q_end; ++q) {
        RecipeAccumulatorState state = staged ? RecipeAccumulatorState{{12, 10, 8}, {13, 11, 9}} : resident;
        if (staged && !first_ring) {
            recipe_checkpoint<17, 18>(state, q, true);
        }
        uint32_t processed = 0;
        for (uint32_t k = 0; k < total_chunks; ++k) {
            const uint32_t origin = (k < local_chunks ? k : k - local_chunks) * 512;
            const uint32_t rows = k < local_chunks ? primary_rows : joint_rows;
            if (origin >= rows) {
                continue;
            }
            recipe_k_tile_offset = 0;
            recipe_k_valid_rows = rows - origin < 512 ? rows - origin : 512;
            const bool last_k = ++processed == valid_chunks;
            sdpa_segment_v2<8, 16, 4, 4, scale, subblock_h, 4, subblock_h, 4, 0, 1, 2, 6, 3, 14, 4, 5, 16, true>(
                state, 1, last_ring && last_k, last_k && (staged || last_ring));
        }
        if (staged && !last_ring) {
            recipe_checkpoint<17, 18>(state, q, false);
        }
        if (!staged) {
            resident = state;
        }
    }
    if (dummy_kv_chunks_for_phase_alignment<false>((q_end - q_begin) * valid_chunks)) {
        CircularBuffer(1).wait_front(64);
        CircularBuffer(1).pop_front(64);
        CircularBuffer(2).wait_front(64);
        CircularBuffer(2).pop_front(64);
    }
}
