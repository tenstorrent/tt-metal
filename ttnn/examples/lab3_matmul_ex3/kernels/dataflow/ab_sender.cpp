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

    uint32_t b_receiver_start_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_receiver_start_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_receiver_end_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_receiver_end_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t b_receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t b_tile_sent_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t b_num_receivers = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t src0_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src1_base_addr = get_arg_val<uint32_t>(arg_idx++);
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
    // Repeat for the second input buffer. next_compile_time_args_offset() is a clean way to
    // determine the index of the next compile-time argument (after TensorAccessorArgs read
    // the number of arguments it required above) without having to hard-code the index.
    constexpr auto src1_layout_args = TensorAccessorArgs<src0_layout_args.next_compile_time_args_offset()>();
    // Finally, construct the address generator for the second input buffer.
    const auto src1_addr_gen = TensorAccessor(src1_layout_args, src1_base_addr, tile_size_bytes_1);

    ////////// SEMAPHORE SETUP //////////
    volatile tt_l1_ptr uint32_t* a_receivers_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_receivers_ready_semaphore_addr);
    volatile tt_l1_ptr uint32_t* a_tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_tile_sent_semaphore_addr);
    volatile tt_l1_ptr uint32_t* b_receivers_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_receivers_ready_semaphore_addr);
    volatile tt_l1_ptr uint32_t* b_tile_sent_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_tile_sent_semaphore_addr);

    // Precompute multicast addresses (these don't change per slab)
    uint64_t a_slab_sent_sem_mcast_addr = get_noc_multicast_addr(
        a_receiver_end_x, a_receiver_end_y, a_receiver_start_x, a_receiver_start_y, a_tile_sent_semaphore_addr);
    uint64_t b_slab_sent_sem_mcast_addr = get_noc_multicast_addr(
        b_receiver_end_x, b_receiver_end_y, b_receiver_start_x, b_receiver_start_y, b_tile_sent_semaphore_addr);

    const uint32_t A_slab_size_bytes = A_slab_tiles * tile_size_bytes_0;
    const uint32_t B_slab_size_bytes = B_slab_tiles * tile_size_bytes_1;

    // Kt dimension is split into K-blocks of size K_block_tiles.
    // For each K-block, read full A and B slabs from DRAM, synchronize with receivers once per slab,
    // then multicast the entire slab.
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        // ``A_slab(b)`` (size: M_block_tiles * K_block_tiles). Read entire slab, then multicast.
        cb_reserve_back(cb_in0, A_slab_tiles);
        uint32_t cb_in0_start_addr = get_write_ptr(cb_in0);
        uint32_t cb_in0_addr = cb_in0_start_addr;
        for (uint32_t slab_a_row = 0; slab_a_row < M_block_tiles; slab_a_row++) {
            uint32_t A_slab_effective_row = tile_offset_row + slab_a_row;
            for (uint32_t slab_a_col = 0; slab_a_col < K_block_tiles; slab_a_col++) {
                uint32_t A_slab_effective_col = b * K_block_tiles + slab_a_col;
                uint32_t a_tile_index = A_slab_effective_row * Kt + A_slab_effective_col;
                noc_async_read_tile(a_tile_index, src0_addr_gen, cb_in0_addr);
                cb_in0_addr += tile_size_bytes_0;
            }
        }
        noc_async_read_barrier();

        noc_semaphore_wait(a_receivers_ready_sem_ptr, a_num_receivers);
        noc_semaphore_set(a_receivers_ready_sem_ptr, 0);

        uint64_t a_slab_mcast_addr = get_noc_multicast_addr(
            a_receiver_end_x, a_receiver_end_y, a_receiver_start_x, a_receiver_start_y, cb_in0_start_addr);
        noc_async_write_multicast(cb_in0_start_addr, a_slab_mcast_addr, A_slab_size_bytes, a_num_receivers);
        noc_async_writes_flushed();

        *a_tile_sent_sem_ptr = VALID;
        noc_semaphore_set_multicast(a_tile_sent_semaphore_addr, a_slab_sent_sem_mcast_addr, a_num_receivers);
        noc_async_write_barrier();

        cb_push_back(cb_in0, A_slab_tiles);

        // ``B_slab(b)`` (size: K_block_tiles * N_block_tiles). Read entire slab, then multicast.
        cb_reserve_back(cb_in1, B_slab_tiles);
        uint32_t cb_in1_start_addr = get_write_ptr(cb_in1);
        uint32_t cb_in1_addr = cb_in1_start_addr;
        for (uint32_t slab_b_row = 0; slab_b_row < K_block_tiles; slab_b_row++) {
            uint32_t B_slab_effective_row = b * K_block_tiles + slab_b_row;
            for (uint32_t slab_b_col = 0; slab_b_col < N_block_tiles; slab_b_col++) {
                uint32_t B_slab_effective_col = tile_offset_col + slab_b_col;
                uint32_t b_tile_index = B_slab_effective_row * Nt + B_slab_effective_col;
                noc_async_read_tile(b_tile_index, src1_addr_gen, cb_in1_addr);
                cb_in1_addr += tile_size_bytes_1;
            }
        }
        noc_async_read_barrier();

        noc_semaphore_wait(b_receivers_ready_sem_ptr, b_num_receivers);
        noc_semaphore_set(b_receivers_ready_sem_ptr, 0);

        uint64_t b_slab_mcast_addr = get_noc_multicast_addr(
            b_receiver_end_x, b_receiver_end_y, b_receiver_start_x, b_receiver_start_y, cb_in1_start_addr);
        noc_async_write_multicast(cb_in1_start_addr, b_slab_mcast_addr, B_slab_size_bytes, b_num_receivers);
        noc_async_writes_flushed();

        *b_tile_sent_sem_ptr = VALID;
        noc_semaphore_set_multicast(b_tile_sent_semaphore_addr, b_slab_sent_sem_mcast_addr, b_num_receivers);
        noc_async_write_barrier();

        cb_push_back(cb_in1, B_slab_tiles);
    }
}
