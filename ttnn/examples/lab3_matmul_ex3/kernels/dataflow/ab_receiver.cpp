// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Read parameters from the kernel's runtime arguments.
    int arg_idx = 0;
    uint32_t a_sender_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_sender_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t a_tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    uint32_t b_sender_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_sender_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t b_tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    const uint32_t Nt = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t Kt = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t M_block_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t N_block_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t K_block_tiles = get_arg_val<uint32_t>(arg_idx++);

    // Offset of the output C_block in the output matrix.
    const uint32_t tile_offset_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_offset_col = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t num_k_blocks = Kt / K_block_tiles;

    // The circular buffers to read the tiles into.
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (which is most often the case).
    constexpr uint32_t tile_size_bytes_0 = get_tile_size(cb_in0);
    constexpr uint32_t tile_size_bytes_1 = get_tile_size(cb_in1);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* a_tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_tile_sent_semaphore_addr);
    volatile tt_l1_ptr uint32_t* b_tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_tile_sent_semaphore_addr);

    // Get the NOC address of the sender's semaphore so we can signal readiness
    uint64_t a_receivers_ready_sem_noc_addr = get_noc_addr(a_sender_x, a_sender_y, a_receivers_ready_semaphore_addr);
    uint64_t b_receivers_ready_sem_noc_addr = get_noc_addr(b_sender_x, b_sender_y, b_receivers_ready_semaphore_addr);

    // Kt dimension is split into K-blocks of size K_block_tiles,
    // such that Kt = num_k_blocks * K_block_tiles.
    // So we have K-block index b in range (0, 1, ..., num_k_blocks-1).
    // Loop over all the K-blocks.
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        // Loop over all the slabs within a K-block, and push them into CB0 and CB1:

        // ``A_slab(b)`` (size: ``M_block_tiles * K_block_tiles``).
        // Order tiles within each slab in the CB in row-major order.
        for (uint32_t slab_a_row = 0; slab_a_row < M_block_tiles; slab_a_row++) {
            // Compute effective row index of the slab within the A matrix
            // based on the offset of the C_block that this core is responsible for,
            // and the index of the current slab.
            uint32_t A_slab_effective_row = tile_offset_row + slab_a_row;
            for (uint32_t slab_a_col = 0; slab_a_col < K_block_tiles; slab_a_col++) {
                // Make sure there is space for one tile in the circular buffer
                cb_reserve_back(cb_in0, 1);

                // Reset a_tile_sent semaphore to INVALID before signaling ready
                noc_semaphore_set(a_tile_sent_sem_ptr, INVALID);

                // Signal a_sender that we're ready to receive the next tile
                noc_semaphore_inc(a_receivers_ready_sem_noc_addr, 1);

                // Wait for a_sender to multicast the tile (semaphore becomes VALID)
                noc_semaphore_wait(a_tile_sent_sem_ptr, VALID);

                // Mark the tile in circular buffer as ready.
                // After this, any kernel (e.g. compute kernel) calling `cb_wait_front` will see this tile.
                cb_push_back(cb_in0, 1);
            }
        }

        // ``B_slab(b)`` (size: ``K_block_tiles * N_block_tiles``).
        // All the indexing logic is equivalent as for A_slab(b) above.
        // Order tiles within each slab in the CB in row-major order.
        for (uint32_t slab_b_row = 0; slab_b_row < K_block_tiles; slab_b_row++) {
            uint32_t B_slab_effective_row = b * K_block_tiles + slab_b_row;
            for (uint32_t slab_b_col = 0; slab_b_col < N_block_tiles; slab_b_col++) {
                // Make sure there is space for one tile in the circular buffer
                cb_reserve_back(cb_in1, 1);

                // Reset b_tile_sent semaphore to INVALID before signaling ready
                noc_semaphore_set(b_tile_sent_sem_ptr, INVALID);

                // Signal b_sender that we're ready to receive the next tile
                noc_semaphore_inc(b_receivers_ready_sem_noc_addr, 1);

                // Wait for b_sender to multicast the tile (semaphore becomes VALID)
                noc_semaphore_wait(b_tile_sent_sem_ptr, VALID);

                // Mark the tile in circular buffer as ready.
                // After this, any kernel (e.g. compute kernel) calling `cb_wait_front` will see this tile.
                cb_push_back(cb_in1, 1);
            }
        }
    }  // K-block loop
}
