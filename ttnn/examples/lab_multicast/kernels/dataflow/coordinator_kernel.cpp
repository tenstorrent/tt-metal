// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Coordinator kernel: reads tiles from DRAM and multicasts them to receiver cores.
// This kernel runs on the sender core (e.g., logical core 0,0).
// Uses double-buffering for improved performance.
void kernel_main() {
    ////////// RUNTIME ARGS //////////
    uint32_t receiver_start_x = get_arg_val<uint32_t>(0);
    uint32_t receiver_start_y = get_arg_val<uint32_t>(1);
    uint32_t receiver_end_x = get_arg_val<uint32_t>(2);
    uint32_t receiver_end_y = get_arg_val<uint32_t>(3);
    uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(4));
    uint32_t tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(5));
    uint32_t src_base_addr = get_arg_val<uint32_t>(6);
    uint32_t n_tiles = get_arg_val<uint32_t>(7);
    uint32_t num_receivers = get_arg_val<uint32_t>(8);

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;  // index=0
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    // Create address generator for the input buffer using TensorAccessorArgs.
    // TensorAccessorArgs extracts data distribution details from compile-time arguments.
    constexpr auto src_layout_args = TensorAccessorArgs<0>();
    const auto src_addr_gen = TensorAccessor(src_layout_args, src_base_addr, tile_size_bytes);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    // Precompute multicast addresses (these don't change per tile)
    uint64_t receiver_sem_mcast_addr = get_noc_multicast_addr(
        receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, tile_sent_semaphore_addr);

    ////////// MAIN LOOP: READ AND MULTICAST EACH TILE //////////
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        // Reserve space in circular buffer (blocking if full - enables double-buffering)
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        // Read tile from DRAM into L1 circular buffer
        noc_async_read_tile(tile_idx, src_addr_gen, l1_write_addr);
        noc_async_read_barrier();

        // Mark tile as ready in CB (for tracking, though we're the only consumer)
        cb_push_back(cb_id_in0, 1);

        // Wait for all receivers to signal they're ready for next tile
        noc_semaphore_wait(sender_sem_ptr, num_receivers);
        noc_semaphore_set(sender_sem_ptr, 0);

        // Get read pointer for the tile we just pushed
        cb_wait_front(cb_id_in0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in0);

        // Multicast tile to all receiver cores
        uint64_t tile_mcast_addr =
            get_noc_multicast_addr(receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, l1_read_addr);
        noc_async_write_multicast(l1_read_addr, tile_mcast_addr, tile_size_bytes, num_receivers);

        // Flush is needed to ensure the multicast command is sent before the semaphore set command
        // because the commands go into separate command buffer FIFOs on some architectures and may not be
        // sent in order they are issued.
        // Note that this doesn't wait on the multicast command to complete, only ensures it is sent.
        noc_async_writes_flushed();

        // Signal receivers that tile has been sent by multicasting VALID to receiver semaphore
        *receiver_sem_ptr = VALID;
        noc_semaphore_set_multicast(tile_sent_semaphore_addr, receiver_sem_mcast_addr, num_receivers);

        // Wait for multicast to complete before freeing CB slot
        noc_async_write_barrier();

        // Free the CB slot for next tile (enables double-buffering)
        cb_pop_front(cb_id_in0, 1);
    }
}
