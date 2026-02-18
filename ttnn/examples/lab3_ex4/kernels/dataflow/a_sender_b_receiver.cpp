// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Read parameters from the kernel's runtime arguments.
    int arg_idx = 0;
    uint32_t a_receiver_start_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_receiver_start_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_receiver_end_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_receiver_end_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t a_receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t a_tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t a_num_receivers = get_arg_val<uint32_t>(arg_idx++);

    uint32_t b_sender_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_sender_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t b_tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    const uint32_t src0_base_addr = get_arg_val<uint32_t>(arg_idx++);
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

    // Create address generators for the input buffers. Address generators can determine
    // physical address based on the provided data layout and base address.
    // Start by extracting the tensor layout parameters from the compile-time arguments.
    // Recall that compile-time arguments are stored as a vector of uint32_t values.
    // TensorAccessorArgs is a clean way to extract the appropriate number of
    // these uint32_t values and store them in an object.
    constexpr auto src0_layout_args = TensorAccessorArgs<0>();
    // Then, construct the address generator for the input buffer.
    // Observe that here we are just constructing the address generator object, but not using it yet.
    // It will be used in the loop below to determine the address to read the tiles from.
    const auto src0_addr_gen = TensorAccessor(src0_layout_args, src0_base_addr, tile_size_bytes_0);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* a_receivers_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_receivers_ready_semaphore_addr);
    volatile tt_l1_ptr uint32_t* a_tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_tile_sent_semaphore_addr);

    const uint32_t A_slab_size_bytes = A_slab_tiles * tile_size_bytes_0;

    // Precompute multicast address that doesn't change per slab.
    uint64_t a_slab_sent_sem_mcast_addr = get_noc_multicast_addr(
        a_receiver_start_x, a_receiver_start_y, a_receiver_end_x, a_receiver_end_y, a_tile_sent_semaphore_addr);

    volatile tt_l1_ptr uint32_t* b_tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_tile_sent_semaphore_addr);

    // Get the NOC address of the sender's semaphore so we can signal readiness
    uint64_t b_receivers_ready_sem_noc_addr = get_noc_addr(b_sender_x, b_sender_y, b_receivers_ready_semaphore_addr);

    // Kt dimension is split into K-blocks of size K_block_tiles,
    // such that Kt = num_k_blocks * K_block_tiles.
    // So we have K-block index b in range (0, 1, ..., num_k_blocks-1).
    // Loop over all the K-blocks.
    // For each K-block, read full A slabs from DRAM and receive full B slabs via multicast.
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        // ``A_slab(b)`` (size: M_block_tiles * K_block_tiles).
        // Order tiles within each slab in the CB in row-major order.
        // Read entire slab from DRAM, then multicast to receivers.
        cb_reserve_back(cb_in0, A_slab_tiles);
        uint32_t cb_in0_start_addr = get_write_ptr(cb_in0);
        uint32_t cb_in0_addr = cb_in0_start_addr;
        for (uint32_t slab_a_row = 0; slab_a_row < M_block_tiles; slab_a_row++) {
            // Compute effective row index of the slab within the A matrix
            // based on the offset of the C_block that this core is responsible for,
            // and the index of the current slab.
            uint32_t A_slab_effective_row = tile_offset_row + slab_a_row;
            for (uint32_t slab_a_col = 0; slab_a_col < K_block_tiles; slab_a_col++) {
                // Compute effective column index of the slab within the A matrix
                // based on the offset of the C_block that this core is responsible for,
                // and the index of the current slab, with K_block_tiles as the stride.
                uint32_t A_slab_effective_col = b * K_block_tiles + slab_a_col;
                // Now that we have effective row and column indices, the rest is a simple
                // 2D to 1D index conversion.
                uint32_t a_tile_index = A_slab_effective_row * Kt + A_slab_effective_col;
                // Recall that src0_addr_gen and src1_addr_gen are address generators for the input buffers.
                // They are used to determine the address to read the tiles from. a_tile_index is the index of the tile
                // to read.
                noc_async_read_tile(a_tile_index, src0_addr_gen, cb_in0_addr);
                cb_in0_addr += tile_size_bytes_0;
            }
        }
        // Wait until the DRAM read is done before waiting on semaphores.
        noc_async_read_barrier();

        noc_semaphore_wait(a_receivers_ready_sem_ptr, a_num_receivers);
        noc_semaphore_set(a_receivers_ready_sem_ptr, 0);

        uint64_t a_slab_mcast_addr = get_noc_multicast_addr(
            a_receiver_start_x, a_receiver_start_y, a_receiver_end_x, a_receiver_end_y, cb_in0_start_addr);
        noc_async_write_multicast(cb_in0_start_addr, a_slab_mcast_addr, A_slab_size_bytes, a_num_receivers);
        noc_async_writes_flushed();

        *a_tile_sent_sem_ptr = VALID;
        noc_semaphore_set_multicast(a_tile_sent_semaphore_addr, a_slab_sent_sem_mcast_addr, a_num_receivers);
        noc_async_write_barrier();

        cb_push_back(cb_in0, A_slab_tiles);

        // ``B_slab(b)`` (size: K_block_tiles * N_block_tiles).
        // Receive entire slab from B sender
        cb_reserve_back(cb_in1, B_slab_tiles);
        noc_semaphore_set(b_tile_sent_sem_ptr, INVALID);
        noc_semaphore_inc(b_receivers_ready_sem_noc_addr, 1);
        noc_semaphore_wait(b_tile_sent_sem_ptr, VALID);
        cb_push_back(cb_in1, B_slab_tiles);
    }
}
