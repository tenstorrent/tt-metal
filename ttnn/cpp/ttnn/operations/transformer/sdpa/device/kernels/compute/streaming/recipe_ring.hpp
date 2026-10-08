// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "recipe_checkpoint.hpp"
#include "../../dataflow/chunked_prefill_utils.hpp"
#ifdef SDPA_RECIPE_RING_CAUSAL
#include "../../q_chunk_remapping.hpp"

// Causal ring attention (is_causal, optionally is_balanced) on one ring step. The step on the device's own K shard
// masks the diagonal in the shard's frame (a balanced shard's two halves keep their order); a balanced ring's
// step on a later device's K skips the early half's Q chunks (the reader sends them nothing), which therefore
// normalize on the last step that reaches them. A balanced step on an earlier device's K sees only that shard's
// early half (the caller clips its valid rows).
struct RecipeRingCausal {
    uint32_t q_chunks;    // per head, as the reader numbers them
    uint32_t heads;
    bool zigzag;          // the reader's balanced Q order
    bool diagonal;        // K is this device's shard: key k is visible to query q iff k <= q
    bool skip_early;      // the early half's Q chunks see none of this step's K
    bool early_last;      // no later step reaches the early half's Q chunks
};
#endif

// QK/PV subblock widths come from the host (recipe_subblock_width in sdpa_recipe.cpp, shared with the dense
// recipe): the largest of 4, 2 and 1 dividing the K chunk / head dim.
#ifndef SDPA_RECIPE_PV_W
#define SDPA_RECIPE_PV_W 4
#endif

template <uint32_t q_tiles, uint32_t scale, uint32_t subblock_h, uint32_t k_tiles, uint32_t d_tiles>
void sdpa_recipe_ring_segment(
    RecipeAccumulatorState& resident,
    uint32_t q_begin,
    uint32_t q_end,
    uint32_t local_chunks,
    uint32_t total_chunks,
    uint32_t primary_rows,
    uint32_t joint_rows,
    bool first_ring,
    bool last_ring
#ifdef SDPA_RECIPE_RING_CAUSAL
    ,
    const RecipeRingCausal& causal
#endif
) {
    static_assert(k_tiles % SDPA_RECIPE_QK_W == 0 && d_tiles % SDPA_RECIPE_PV_W == 0);
    const bool staged = q_end - q_begin > 1;
    constexpr uint32_t chunk_rows = k_tiles * 32;
    recipe_k_chunk_rows = chunk_rows;
    const uint32_t valid_chunks =
        (primary_rows + chunk_rows - 1) / chunk_rows + (joint_rows + chunk_rows - 1) / chunk_rows;
    ASSERT(valid_chunks > 0);
#ifdef SDPA_RECIPE_RING_CAUSAL
    uint32_t pushed_chunks = 0;
#endif
    for (uint32_t q = q_begin; q < q_end; ++q) {
#ifdef SDPA_RECIPE_RING_CAUSAL
        const uint32_t q_chunk = decompose_global_q_index(q, causal.q_chunks, causal.heads, causal.zigzag).q_chunk;
        const bool early = q_chunk < causal.q_chunks / 2;
        if (early && causal.skip_early) {
            continue;
        }
        pushed_chunks += valid_chunks;
        const bool q_last_ring = last_ring || (early && causal.early_last);
        // On the diagonal, K chunks starting past the Q chunk's last row are masked whole: popped unread.
        const uint32_t k_end = causal.diagonal ? (q_chunk * q_tiles + q_tiles - 1) / k_tiles + 1 : total_chunks;
        const uint32_t live_chunks = valid_chunks < k_end ? valid_chunks : k_end;
#else
        const bool q_last_ring = last_ring;
        const uint32_t live_chunks = valid_chunks;
#endif
        RecipeAccumulatorState state = staged ? RecipeAccumulatorState{{12, 10, 8}, {13, 11, 9}} : resident;
        if (staged && !first_ring) {
#ifdef SDPA_RING_STREAM_STATE
            recipe_checkpoint_restore_stream<q_tiles, 17, 18>(state, q);
#else
            recipe_checkpoint<q_tiles, 17, 18, d_tiles>(state, q, true);
#endif
        }
        uint32_t processed = 0;
        for (uint32_t k = 0; k < total_chunks; ++k) {
            const uint32_t origin = (k < local_chunks ? k : k - local_chunks) * chunk_rows;
            const uint32_t rows = k < local_chunks ? primary_rows : joint_rows;
            if (origin >= rows) {
                continue;
            }
#ifdef SDPA_RECIPE_RING_CAUSAL
            if (k >= k_end) {
                CircularBuffer(1).wait_front(k_tiles * d_tiles);
                CircularBuffer(1).pop_front(k_tiles * d_tiles);
                CircularBuffer(2).wait_front(k_tiles * d_tiles);
                CircularBuffer(2).pop_front(k_tiles * d_tiles);
                continue;
            }
            recipe_causal_tile_delta = static_cast<int32_t>(k * k_tiles) - static_cast<int32_t>(q_chunk * q_tiles);
            recipe_causal_edge = causal.diagonal && recipe_causal_tile_delta + static_cast<int32_t>(k_tiles) > 0;
#endif
            recipe_k_tile_offset = 0;
            recipe_k_valid_rows = rows - origin < chunk_rows ? rows - origin : chunk_rows;
            const bool last_k = ++processed == live_chunks;
#ifdef SDPA_RING_STREAM_STATE
            if (staged && !q_last_ring && last_k) {
                recipe_stream_save_slot = q;  // the fused chunk hands its finished O rows to the writer
                recipe_stream_saved_rows = 0;
            }
#endif
            sdpa_segment_v2<
                q_tiles,
                k_tiles,
                d_tiles,
                d_tiles,
                scale,
                subblock_h,
                SDPA_RECIPE_QK_W,
                subblock_h,
                SDPA_RECIPE_PV_W,
                0,
                1,
                2,
                6,
                3,
                14,
                4,
                5,
                16,
                true>(
                state, 1, q_last_ring && last_k, last_k && (staged || q_last_ring));
        }
        if (staged && !q_last_ring) {
#ifdef SDPA_RING_STREAM_STATE
            // The next block of the first ring iteration reuses the banks without a restore.
            recipe_checkpoint_save_tail<q_tiles, 17, 18>(state, q, first_ring && q + 1 < q_end);
#else
            recipe_checkpoint<q_tiles, 17, 18, d_tiles>(state, q, false);
#endif
        }
        if (!staged) {
            resident = state;
        }
    }
#ifdef SDPA_RECIPE_RING_CAUSAL
    if (dummy_kv_chunks_for_phase_alignment<false>(pushed_chunks)) {
#else
    if (dummy_kv_chunks_for_phase_alignment<false>((q_end - q_begin) * valid_chunks)) {
#endif
        CircularBuffer(1).wait_front(k_tiles * d_tiles);
        CircularBuffer(1).pop_front(k_tiles * d_tiles);
        CircularBuffer(2).wait_front(k_tiles * d_tiles);
        CircularBuffer(2).pop_front(k_tiles * d_tiles);
    }
}
