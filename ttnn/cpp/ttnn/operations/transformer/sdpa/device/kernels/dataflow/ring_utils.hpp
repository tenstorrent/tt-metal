// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ring SDPA topology helpers shared between compute and dataflow kernels.
// Contains the single-source-of-truth sequential ring_id assignment state machine
// used by RingSDPAOpIndexer, RingSDPAOpReceiver, and pre-scan helpers.

#pragma once

#include <cstdint>

/**
 * Sequential single-direction ring_id sequencer.
 * Computes which device's KV shard to process at each ring iteration.
 * Iteration 0 is always the local device (ring_index).
 * Subsequent iterations step in a fixed direction determined at construction.
 *
 * direction 0 = backward: ring_id increments each step (ring_index, ring_index+1, ...)
 * direction 1 = forward:  ring_id decrements each step (ring_index, ring_index-1, ...)
 *
 * The get_next_ring_id() method accepts a sync callback, allowing callers to
 * inject hardware synchronization (e.g. semaphore waits) at each iteration.
 * Signature: void(uint32_t dir, uint32_t wait_val)
 *   - dir:      fixed direction for this sequencer
 *   - wait_val: transfer_idx at the time of the call (0 on first iteration → no wait)
 * Pass a no-op lambda for sync-free usage (compute kernel, pre-scan).
 */
struct RingIdSequencer {
    uint32_t ring_index = 0;
    uint32_t ring_size = 0;
    uint32_t direction = 0;  // 0 = backward, 1 = forward
    int32_t step = 1;        // +1 for backward, -1 for forward
    uint32_t transfer_idx = 0;

    RingIdSequencer() = default;

    RingIdSequencer(uint32_t ring_index_, uint32_t ring_size_, uint32_t direction_) :
        ring_index(ring_index_),
        ring_size(ring_size_),
        direction(direction_),
        step(direction_ == 0 ? 1 : -1),
        transfer_idx(0) {}

    template <typename SyncFn>
    uint32_t get_next_ring_id(SyncFn&& sync_fn) {
        // ring_size * ring_size ensures a non-negative value before modulo for any valid input
        uint32_t ring_id =
            ((int32_t)ring_index + (int32_t)transfer_idx * step + (int32_t)(ring_size * ring_size)) % ring_size;

        sync_fn(direction, transfer_idx);

        transfer_idx++;
        return ring_id;
    }
};

/**
 * Find the last ring iteration that performs actual KV computation.
 * Creates a copy of the sequencer and iterates it with no synchronization.
 *
 * @param seq               Sequencer state (copied — original is not modified)
 * @param local_padded_Nt   Per-device padded sequence length in tiles
 * @param global_n_tile_id  Logical (unpadded) sequence length in tiles (logical_n / TILE_HEIGHT)
 * @param L                 Joint sequence length in elements (0 if no joint attention)
 */
inline uint32_t find_last_active_ring_iter(
    RingIdSequencer seq, uint32_t local_padded_Nt, uint32_t global_n_tile_id, uint32_t L) {
    uint32_t last_active = 0;
    auto no_sync = [](uint32_t, uint32_t) {};

    for (uint32_t t = 0; t < seq.ring_size; ++t) {
        uint32_t ring_id = seq.get_next_ring_id(no_sync);
        bool does_joint = (ring_id == seq.ring_size - 1);
        uint32_t kv_start = ring_id * local_padded_Nt;
        bool does_work = (kv_start <= global_n_tile_id) || (does_joint && L != 0);
        if (does_work) {
            last_active = t;
        }
    }

    return last_active;
}
