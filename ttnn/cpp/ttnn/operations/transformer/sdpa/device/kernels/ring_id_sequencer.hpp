// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pure ring_id assignment state machine shared by host planning and kernels.

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
 * computation and the direction switch. Host planning passes a no-op callback.
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
                // Receiving from forward direction -> go backwards
                sender_ring_id = (ring_index - received[curr_dir] + ring_size) % ring_size;
                sync_wait_val = received[curr_dir];
            } else {
                // Receiving from backward direction -> go forwards
                sender_ring_id = (ring_index + received[curr_dir]) % ring_size;
                sync_wait_val = received[curr_dir] + 1;
            }
        }

        sync_fn(sync_dir, sync_wait_val);

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
