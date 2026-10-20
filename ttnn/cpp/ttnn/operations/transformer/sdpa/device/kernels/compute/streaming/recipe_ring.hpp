// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "recipe_checkpoint.hpp"
#include "recipe_config.hpp"
#include "../../dataflow/chunked_prefill_utils.hpp"
#include "../../q_chunk_remapping.hpp"

// Causal ring attention (recipe::ring_causal: is_causal, optionally is_balanced) on one ring step. The step on the
// device's own K shard masks the diagonal in the shard's frame (a balanced shard's two halves keep their order); a
// balanced ring's step on a later device's K skips the early half's Q chunks (the reader sends them nothing), which
// therefore normalize on the last step that reaches them. A balanced step on an earlier device's K sees only that
// shard's early half (the caller clips its valid rows).
struct RecipeRingCausal {
    uint32_t q_chunks;  // per head, as the reader numbers them
    uint32_t heads;
    bool zigzag;      // the reader's balanced Q order
    bool diagonal;    // K is this device's shard: key k is visible to query q iff k <= q
    bool skip_early;  // the early half's Q chunks see none of this step's K
    bool early_last;  // no later step reaches the early half's Q chunks
};

// Ring chunked prefill (recipe::ring_chunked; Q is the current chunk group's slab, K/V the cache of every group so
// far): every step masks in the sequence's frame (recipe_tail.hpp: recipe_chunked_k_tile). The reader sends a K chunk
// while its first global tile is before logical_nt (the first chunk) or live_end_nt (the rest: the device's last Q row
// with the dense causal skip, else logical_nt).
struct RecipeRingChunkedCausal {
    uint32_t q_chunks;
    uint32_t heads;
    bool zigzag;
    bool diagonal;
    bool skip_early;
    bool early_last;
    uint32_t q_tile0;  // global tile of this device's first Q row
    uint32_t ring_id;  // the K shard's device
    uint32_t logical_nt;
    uint32_t live_end_nt;
    uint32_t local_tiles;  // K tiles per device
};

// A K/V chunk the reader sent that compute does not read (fully masked, or phase-alignment padding).
template <uint32_t k_chunk_tiles, uint32_t v_chunk_tiles>
ALWI void recipe_ring_pop_chunk() {
    CircularBuffer(1).wait_front(k_chunk_tiles);
    CircularBuffer(1).pop_front(k_chunk_tiles);
    CircularBuffer(2).wait_front(v_chunk_tiles);
    CircularBuffer(2).pop_front(v_chunk_tiles);
}

// Chunked prefill: whether the reader sends the K chunk starting at local tile `tile` (ring_joint_reader_impl.hpp:
// chunked_kv_chunk_is_live).
template <typename Causal>
ALWI bool recipe_ring_chunk_sent(const Causal& causal, uint32_t tile, bool first) {
    return tile < causal.local_tiles &&
           recipe_chunked_k_tile(causal.ring_id, tile) < (first ? causal.logical_nt : causal.live_end_nt);
}

// QK/PV subblock widths (recipe::qk_w / pv_w) come from the host (recipe_subblock_width in sdpa_recipe.cpp, shared
// with the dense recipe): the largest of 4, 2 and 1 dividing the K chunk / head dim.

// d_tiles: Q/K head dim; vd_tiles: V and output head dim (MLA: vd_tiles < d_tiles). Causal ring steps pass their
// RecipeRingCausal (chunked prefill: RecipeRingChunkedCausal) as `causal`; other builds pass none.
template <
    uint32_t q_tiles,
    uint32_t scale,
    uint32_t subblock_h,
    uint32_t k_tiles,
    uint32_t d_tiles,
    uint32_t vd_tiles,
    typename... Causal>
void sdpa_recipe_ring_segment(
    RecipeAccumulatorState& resident,
    uint32_t q_begin,
    uint32_t q_end,
    uint32_t local_chunks,
    uint32_t total_chunks,
    uint32_t primary_rows,
    uint32_t joint_rows,
    bool first_ring,
    bool last_ring,
    uint32_t padded_slots,
    const Causal&... causal) {
    static_assert(sizeof...(Causal) == (recipe::ring_causal ? 1 : 0));
    static_assert(k_tiles % recipe::qk_w == 0 && vd_tiles % recipe::pv_w == 0);
    const bool staged = q_end - q_begin > 1;
    constexpr uint32_t chunk_rows = k_tiles * 32;
    recipe_k_chunk_rows = chunk_rows;
    const uint32_t valid_chunks =
        (primary_rows + chunk_rows - 1) / chunk_rows + (joint_rows + chunk_rows - 1) / chunk_rows;
    ASSERT(valid_chunks > 0);
    uint32_t pushed_chunks = 0;  // ring causal
    for (uint32_t q = q_begin; q < q_end; ++q) {
        bool q_last_ring = last_ring;
        uint32_t live_chunks = valid_chunks;
        [[maybe_unused]] uint32_t q_chunk = 0;
        [[maybe_unused]] uint32_t q_first = 0;
        [[maybe_unused]] uint32_t k_end = total_chunks;
        if constexpr (recipe::ring_causal) {
            const auto& c = (causal, ...);
            q_chunk = decompose_global_q_index(q, c.q_chunks, c.heads, c.zigzag).q_chunk;
            const bool early = q_chunk < c.q_chunks / 2;
            if (early && c.skip_early) {
                continue;
            }
            q_last_ring = last_ring || (early && c.early_last);
            if constexpr (recipe::ring_chunked != 0) {
                // K chunks the reader sends, and the ones with a key some row of this Q chunk sees (a prefix of them:
                // global K positions grow with the local index); the rest are popped unread.
                q_first = c.q_tile0 + q_chunk * q_tiles;
                live_chunks = 0;
                for (uint32_t k = 0; k < local_chunks; ++k) {
                    if (recipe_ring_chunk_sent(c, k * k_tiles, k == 0)) {
                        ++pushed_chunks;
                        live_chunks += recipe_chunked_k_tile(c.ring_id, k * k_tiles) < q_first + q_tiles ? 1 : 0;
                    }
                }
            } else {
                pushed_chunks += valid_chunks;
                // On the diagonal, K chunks starting past the Q chunk's last row are masked whole: popped unread.
                k_end = c.diagonal ? (q_chunk * q_tiles + q_tiles - 1) / k_tiles + 1 : total_chunks;
                live_chunks = valid_chunks < k_end ? valid_chunks : k_end;
            }
        }
        RecipeAccumulatorState state = staged ? RecipeAccumulatorState{{12, 10, 8}, {13, 11, 9}} : resident;
        if (staged && !first_ring) {
            if constexpr (recipe::ring_stream_state) {
                recipe_checkpoint_restore_stream<q_tiles, 17, 18>(state, q);
            } else {
                recipe_checkpoint<q_tiles, 17, 18, vd_tiles>(state, q, true);
            }
        }
        uint32_t processed = 0;
        for (uint32_t k = 0; k < total_chunks; ++k) {
            const uint32_t origin = (k < local_chunks ? k : k - local_chunks) * chunk_rows;
            const uint32_t rows = k < local_chunks ? primary_rows : joint_rows;
            if (origin >= rows) {
                continue;
            }
            if constexpr (recipe::ring_causal && recipe::ring_chunked != 0) {
                const auto& c = (causal, ...);
                if (!recipe_ring_chunk_sent(c, k * k_tiles, k == 0)) {
                    continue;
                }
                // Masked when its last key reaches the Q chunk's first row; whole when its first key is past the last.
                const uint32_t k_last = (k + 1) * k_tiles < c.local_tiles ? (k + 1) * k_tiles : c.local_tiles;
                if (recipe_chunked_k_tile(c.ring_id, k * k_tiles) >= q_first + q_tiles) {
                    recipe_ring_pop_chunk<k_tiles * d_tiles, k_tiles * vd_tiles>();
                    continue;
                }
                recipe_causal_k_tile0 = k * k_tiles;
                recipe_causal_ring_id = c.ring_id;
                recipe_causal_tile_delta = -static_cast<int32_t>(q_first);
                recipe_causal_edge = recipe_chunked_k_tile(c.ring_id, k_last - 1) >= q_first;
            } else if constexpr (recipe::ring_causal) {
                const auto& c = (causal, ...);
                if (k >= k_end) {
                    recipe_ring_pop_chunk<k_tiles * d_tiles, k_tiles * vd_tiles>();
                    continue;
                }
                recipe_causal_tile_delta = static_cast<int32_t>(k * k_tiles) - static_cast<int32_t>(q_chunk * q_tiles);
                recipe_causal_edge = c.diagonal && recipe_causal_tile_delta + static_cast<int32_t>(k_tiles) > 0;
            }
            recipe_k_tile_offset = 0;
            recipe_k_valid_rows = rows - origin < chunk_rows ? rows - origin : chunk_rows;
            const bool last_k = ++processed == live_chunks;
            if constexpr (recipe::ring_stream_state) {
                if (staged && !q_last_ring && last_k) {
                    recipe_stream_save_slot = q;  // the fused chunk hands its finished O rows to the writer
                    recipe_stream_saved_rows = 0;
                }
            }
            sdpa_segment_v2<
                q_tiles,
                k_tiles,
                d_tiles,
                vd_tiles,
                scale,
                subblock_h,
                recipe::qk_w,
                subblock_h,
                recipe::pv_w,
                0,
                1,
                2,
                6,
                3,
                14,
                4,
                5,
                16,
                true>(state, 1, q_last_ring && last_k, last_k && (staged || q_last_ring));
        }
        if (staged && !q_last_ring) {
            if constexpr (recipe::ring_stream_state) {
                // The next block of the first ring iteration reuses the banks without a restore.
                recipe_checkpoint_save_tail<q_tiles, 17, 18>(state, q, first_ring && q + 1 < q_end);
            } else {
                recipe_checkpoint<q_tiles, 17, 18, vd_tiles>(state, q, false);
            }
        }
        if (!staged) {
            resident = state;
        }
    }
    // Padded multicast slots: the reader runs its multicast group's longest Q count and pushes the K/V chunks each
    // padded slot receives, as for a real Q chunk (none for one it skips), so this core's CBs advance with the
    // injector's. Pop them unread.
    uint32_t padded_chunks = 0;
    if (padded_slots > 0) {
        uint32_t chunks_per_q = valid_chunks;
        if constexpr (recipe::ring_causal && recipe::ring_chunked != 0) {
            const auto& c = (causal, ...);
            chunks_per_q = 0;
            for (uint32_t k = 0; k < local_chunks; ++k) {
                chunks_per_q += recipe_ring_chunk_sent(c, k * k_tiles, k == 0) ? 1 : 0;
            }
        }
        for (uint32_t slot = 0; slot < padded_slots; ++slot) {
            if constexpr (recipe::ring_causal) {
                const auto& c = (causal, ...);
                const uint32_t slot_q_chunk =
                    decompose_global_q_index(q_end + slot, c.q_chunks, c.heads, c.zigzag).q_chunk;
                if (c.skip_early && slot_q_chunk < c.q_chunks / 2) {
                    continue;
                }
            }
            padded_chunks += chunks_per_q;
        }
    }
    for (uint32_t i = 0; i < padded_chunks; ++i) {
        recipe_ring_pop_chunk<k_tiles * d_tiles, k_tiles * vd_tiles>();
    }
    const uint32_t sent_chunks = recipe::ring_causal ? pushed_chunks : (q_end - q_begin) * valid_chunks;
    if (dummy_kv_chunks_for_phase_alignment<false>(sent_chunks + padded_chunks)) {
        recipe_ring_pop_chunk<k_tiles * d_tiles, k_tiles * vd_tiles>();
    }
}
