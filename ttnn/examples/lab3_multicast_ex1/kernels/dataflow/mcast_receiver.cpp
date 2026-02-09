// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Multicast receiver kernel: receives multicast tiles from sender core.
// This kernel runs on receiver cores (e.g., logical cores 1,0 through 3,0).
// It receives tiles into the input circular buffer (c_0) where they will be
// consumed by the compute kernel.
void kernel_main() {
    ////////// RUNTIME ARGS //////////
    uint32_t sender_x = get_arg_val<uint32_t>(0);
    uint32_t sender_y = get_arg_val<uint32_t>(1);
    uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(2));
    uint32_t tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(3));
    uint32_t n_tiles = get_arg_val<uint32_t>(4);

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;  // index=0

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    // Get the NOC address of the sender's semaphore so we can signal readiness
    uint64_t receivers_ready_sem_noc_addr = get_noc_addr(sender_x, sender_y, receivers_ready_semaphore_addr);

    ////////// MAIN LOOP: RECEIVE EACH TILE //////////
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        // Reserve space in input CB for incoming tile
        cb_reserve_back(cb_id_in0, 1);

        // Reset tile_sent semaphore to INVALID before signaling ready
        noc_semaphore_set(tile_sent_sem_ptr, INVALID);

        // Signal sender that we're ready to receive the next tile
        noc_semaphore_inc(receivers_ready_sem_noc_addr, 1);

        // Wait for sender to multicast the tile (semaphore becomes VALID)
        noc_semaphore_wait(tile_sent_sem_ptr, VALID);

        // Tile has been received into our L1 at the CB write pointer.
        // Mark it as available in the input CB for the compute kernel to consume.
        cb_push_back(cb_id_in0, 1);
    }
}
