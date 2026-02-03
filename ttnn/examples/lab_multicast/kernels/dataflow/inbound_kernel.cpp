// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Helper function to copy a tile from one CB to another CB (e.g., input CB to output CB) via L1.
inline void copy_tile_between_cb(uint32_t src_addr, uint32_t dst_addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t i = 0; i < bytes / sizeof(uint32_t); i++) {
        dst[i] = src[i];
    }
}

// Inbound kernel: receives multicast tiles from coordinator core.
// This kernel runs on receiver cores (e.g., logical cores 1,0 through 3,0).
// Uses double-buffering for improved performance.
void kernel_main() {
    ////////// RUNTIME ARGS //////////
    uint32_t sender_x = get_arg_val<uint32_t>(0);
    uint32_t sender_y = get_arg_val<uint32_t>(1);
    uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(2));
    uint32_t tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(3));
    uint32_t n_tiles = get_arg_val<uint32_t>(4);

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;   // index=0
    constexpr uint32_t cb_id_out0 = tt::CB::c_out0; // index=16
    uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    // Get the NOC address of the sender's semaphore so we can signal readiness
    uint64_t sender_sem_noc_addr = get_noc_addr(sender_x, sender_y, receivers_ready_semaphore_addr);

    ////////// MAIN LOOP: RECEIVE EACH TILE //////////
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        // Reserve space in input CB for incoming tile
        cb_reserve_back(cb_id_in0, 1);

        // Reset our receiver semaphore to INVALID before signaling ready
        noc_semaphore_set(receiver_sem_ptr, INVALID);

        // Signal coordinator that we're ready to receive the next tile
        noc_semaphore_inc(sender_sem_noc_addr, 1);

        // Wait for coordinator to multicast the tile (semaphore becomes VALID)
        noc_semaphore_wait(receiver_sem_ptr, VALID);

        // Tile has been received into our L1 at the CB write pointer
        // Mark it as available in the input CB
        cb_push_back(cb_id_in0, 1);

        // Wait for tile to be available and get its address
        cb_wait_front(cb_id_in0, 1);

        // Reserve space in output CB and copy the tile
        cb_reserve_back(cb_id_out0, 1);
        copy_tile_between_cb(get_read_ptr(cb_id_in0), get_write_ptr(cb_id_out0), tile_size_bytes);
        cb_push_back(cb_id_out0, 1);

        // Free the input CB slot for next tile
        cb_pop_front(cb_id_in0, 1);
    }
}
