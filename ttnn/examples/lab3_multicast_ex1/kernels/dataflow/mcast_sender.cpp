// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Multicast sender kernel: reads tiles from DRAM and multicasts them to receiver cores.
// This kernel runs on the sender core (e.g., logical core 0,0).
// Sends tiles_per_batch tiles at a time, using double-buffering for improved performance.
void kernel_main() {
    ////////// COMPILE-TIME ARGS //////////
    constexpr auto src_layout_args = TensorAccessorArgs<0>();
    constexpr uint32_t tiles_per_batch = get_compile_time_arg_val(src_layout_args.next_compile_time_args_offset());

    ////////// RUNTIME ARGS //////////
    int arg_idx = 0;
    uint32_t receiver_start_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_start_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_end_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_end_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t src_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t n_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_receivers = get_arg_val<uint32_t>(arg_idx++);

    ////////// BUFFER SETUP //////////
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);
    const uint32_t batch_size_bytes = tiles_per_batch * tile_size_bytes;

    // Create address generator for the input buffer using TensorAccessorArgs.
    // TensorAccessorArgs extracts data distribution details from compile-time arguments.
    const auto src_addr_gen = TensorAccessor(src_layout_args, src_base_addr, tile_size_bytes);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* receivers_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);
    volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

    // Precompute multicast addresses (these don't change per tile)
    uint64_t tile_sent_mcast_addr = get_noc_multicast_addr(
        receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, tile_sent_semaphore_addr);

    ////////// MAIN LOOP: READ AND MULTICAST EACH BATCH OF TILES //////////
    const uint32_t num_batches = n_tiles / tiles_per_batch;
    for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        uint32_t batch_start_tile = batch_idx * tiles_per_batch;

        // Reserve space for full batch (blocking if full - enables double-buffering)
        cb_reserve_back(cb_id_in0, tiles_per_batch);
        uint32_t cb_write_addr = get_write_ptr(cb_id_in0);

        // Read all tiles in batch from DRAM into L1
        for (uint32_t i = 0; i < tiles_per_batch; i++) {
            noc_async_read_tile(batch_start_tile + i, src_addr_gen, cb_write_addr);
            cb_write_addr += tile_size_bytes;
        }
        noc_async_read_barrier();

        // Mark batch as ready in CB (for tracking, though we're the only consumer)
        cb_push_back(cb_id_in0, tiles_per_batch);

        // Wait for all receivers to signal they're ready for next batch
        noc_semaphore_wait(receivers_ready_sem_ptr, num_receivers);
        noc_semaphore_set(receivers_ready_sem_ptr, 0);

        // Get read pointer for the batch we just pushed
        cb_wait_front(cb_id_in0, tiles_per_batch);
        uint32_t cb_read_addr = get_read_ptr(cb_id_in0);

        // Multicast entire batch to all receiver cores
        uint64_t batch_mcast_addr =
            get_noc_multicast_addr(receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, cb_read_addr);
        noc_async_write_multicast(cb_read_addr, batch_mcast_addr, batch_size_bytes, num_receivers);

        // Flush is needed to ensure the multicast command is sent before the semaphore set command
        // because the commands go into separate command buffer FIFOs on some architectures and may not be
        // sent in order they are issued.
        // Note that this doesn't wait on the multicast command to complete, only ensures it is sent.
        noc_async_writes_flushed();

        // Signal receivers that batch has been sent by multicasting VALID to receiver semaphore
        *tile_sent_sem_ptr = VALID;
        noc_semaphore_set_multicast(tile_sent_semaphore_addr, tile_sent_mcast_addr, num_receivers);

        // Wait for multicast to complete before freeing CB slot
        noc_async_write_barrier();

        // Free the CB slot for next batch (enables double-buffering)
        cb_pop_front(cb_id_in0, tiles_per_batch);
    }
}
