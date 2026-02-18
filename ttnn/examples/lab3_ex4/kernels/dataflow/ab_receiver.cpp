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
    const uint32_t A_slab_tiles = M_block_tiles * K_block_tiles;
    const uint32_t B_slab_tiles = K_block_tiles * N_block_tiles;

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

    // Kt dimension is split into K-blocks. For each K-block, receive full A and B slabs via multicast.
    // Reserve space for entire slab, signal ready once, wait for sender once, then push entire slab.
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        // ``A_slab(b)``: receive entire slab from A sender
        cb_reserve_back(cb_in0, A_slab_tiles);
        noc_semaphore_set(a_tile_sent_sem_ptr, INVALID);
        noc_semaphore_inc(a_receivers_ready_sem_noc_addr, 1);
        noc_semaphore_wait(a_tile_sent_sem_ptr, VALID);
        cb_push_back(cb_in0, A_slab_tiles);

        // ``B_slab(b)``: receive entire slab from B sender
        cb_reserve_back(cb_in1, B_slab_tiles);
        noc_semaphore_set(b_tile_sent_sem_ptr, INVALID);
        noc_semaphore_inc(b_receivers_ready_sem_noc_addr, 1);
        noc_semaphore_wait(b_tile_sent_sem_ptr, VALID);
        cb_push_back(cb_in1, B_slab_tiles);
    }
}
