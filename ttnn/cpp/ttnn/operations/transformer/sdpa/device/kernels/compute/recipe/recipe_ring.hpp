// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Ring continuation of the recipe core loop (ring_joint_sdpa.cpp): one Q chunk's K chunks of one ring iteration,
// continuing its recurrent state. A core with one Q chunk keeps the state in L1 across the ring (resident); a core
// with several checkpoints each Q chunk's state through the writer between iterations (staged, recipe_checkpoint.hpp).
#include "recipe_checkpoint.hpp"
#include "recipe_config.hpp"
#include "recipe_tail.hpp"
#include "../../recipe_ring_cbs.hpp"

// A K/V chunk the reader sent that compute does not read (fully masked, or multicast / phase-alignment padding).
template <uint32_t k_chunk_tiles, uint32_t v_chunk_tiles>
ALWI void recipe_ring_pop_chunk() {
    CircularBuffer(recipe_ring_cb::k).wait_front(k_chunk_tiles);
    CircularBuffer(recipe_ring_cb::k).pop_front(k_chunk_tiles);
    CircularBuffer(recipe_ring_cb::v).wait_front(v_chunk_tiles);
    CircularBuffer(recipe_ring_cb::v).pop_front(v_chunk_tiles);
}

// One Q chunk's state for one ring iteration: the state banks (resident: `state` carries over from the Q chunk's
// previous iteration; staged: the banks reset, restored from the writer unless this is the Q chunk's first iteration),
// then `chunks` K chunks through `step(state, chunk, last_k)`, then a save (staged, unless the Q chunk normalized).
// `step` sets the chunk's mask and runs the recipe step. One state object, updated in place: the ring kernel's frame
// lies under the whole recipe step on the TRISC stacks (Wormhole: a few hundred bytes).
template <uint32_t q_tiles, uint32_t vd_tiles, typename Step>
ALWI void recipe_ring_q_block(
    RecipeAccumulatorState& state,
    bool staged,
    bool first_iteration,
    bool last_iteration,
    uint32_t slot,
    bool next_block_unrestored,
    uint32_t chunks,
    const Step& step) {
    using namespace recipe_ring_cb;
    if (staged) {
        state.prev = {sum_a, max_a, out_a};
        state.cur = {sum_b, max_b, out_b};
        state.processed_chunks = 0;
    }
    if (staged && !first_iteration) {
        if constexpr (recipe::ring_stream_state) {
            recipe_checkpoint_restore_stream<q_tiles, checkpoint_request, checkpoint_ack>(state, slot);
        } else {
            recipe_checkpoint<q_tiles, checkpoint_request, checkpoint_ack, vd_tiles>(state, slot, true);
        }
    }
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        const bool last_k = chunk + 1 == chunks;
        if constexpr (recipe::ring_stream_state) {
            if (staged && !last_iteration && last_k) {
                recipe_stream_save_slot = slot;  // the fused chunk hands its finished O rows to the writer
                recipe_stream_saved_rows = 0;
            }
        }
        step(state, chunk, last_k);
    }
    if (staged && !last_iteration) {
        if constexpr (recipe::ring_stream_state) {
            // A block that starts without a restore (the first ring iteration) reuses the banks at once.
            recipe_checkpoint_save_tail<q_tiles, checkpoint_request, checkpoint_ack>(
                state, slot, next_block_unrestored);
        } else {
            recipe_checkpoint<q_tiles, checkpoint_request, checkpoint_ack, vd_tiles>(state, slot, false);
        }
    }
}
