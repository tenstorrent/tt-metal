// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ring SDPA topology helpers shared between compute and dataflow kernels.
// Contains the single-source-of-truth ring_id assignment state machine
// used by RingSDPAOpIndexer, RingSDPAOpReceiver, and pre-scan helpers.

#pragma once

#include <cstdint>

/**
 * Direction-alternating ring_id sequencer.
 * Computes which device's KV shard to process at each ring iteration.
 * Iteration 0 is always the local device (ring_index).
 * Subsequent iterations alternate between backward and forward directions.
 *
 * The get_next_ring_id() method accepts a sync callback, allowing callers to
 * inject hardware synchronization (e.g. semaphore waits) between the ring_id
 * computation and the direction switch. This is the single implementation of
 * the ring_id assignment logic — both RingSDPAOpIndexer and RingSDPAOpReceiver
 * delegate to it.
 */
struct RingIdSequencer {
    uint32_t ring_index = 0;
    uint32_t ring_size = 0;
    uint32_t received[2] = {0, 0};  // [backward, forward]
    uint32_t expected[2] = {0, 0};  // [backward, forward]
    uint32_t curr_dir = 0;          // 0 = backward, 1 = forward
    uint32_t transfer_idx = 0;

    RingIdSequencer() = default;

    RingIdSequencer(uint32_t ring_index_, uint32_t ring_size_, uint32_t backward_expected, uint32_t forward_expected) :
        ring_index(ring_index_),
        ring_size(ring_size_),
        received{0, 0},
        expected{backward_expected, forward_expected},
        curr_dir(0),
        transfer_idx(0) {}

    /**
     * Compute the next ring_id and advance the state machine.
     *
     * @param sync_fn  Callback invoked between ring_id computation and direction switch.
     *                 Signature: void(uint32_t dir, uint32_t wait_val)
     *                 - dir: current direction index (for semaphore selection)
     *                 - wait_val: semaphore threshold (0 on first iteration)
     *                 Pass a no-op lambda for sync-free usage (compute kernel, pre-scan).
     */
    template <typename SyncFn>
    uint32_t get_next_ring_id(SyncFn&& sync_fn) {
        uint32_t sender_ring_id;
        uint32_t sync_dir = curr_dir;
        uint32_t sync_wait_val;

        if (transfer_idx == 0) {
            sender_ring_id = ring_index;
            sync_wait_val = 0;
        } else {
            received[curr_dir] += 1;
            if (curr_dir == 1) {
                // Receiving from forward direction → go backwards
                sender_ring_id = (ring_index - received[curr_dir] + ring_size) % ring_size;
                sync_wait_val = received[curr_dir];
            } else {
                // Receiving from backward direction → go forwards
                sender_ring_id = (ring_index + received[curr_dir]) % ring_size;
                sync_wait_val = received[curr_dir] + 1;
            }
        }

        sync_fn(sync_dir, sync_wait_val);

        // Direction switch
        if (transfer_idx == 0) {
            if (expected[curr_dir] == 0) {
                curr_dir = 1 - curr_dir;
            }
        } else {
            uint32_t next_dir = 1 - curr_dir;
            if (received[next_dir] < expected[next_dir]) {
                curr_dir = next_dir;
            }
        }

        transfer_idx++;
        return sender_ring_id;
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
