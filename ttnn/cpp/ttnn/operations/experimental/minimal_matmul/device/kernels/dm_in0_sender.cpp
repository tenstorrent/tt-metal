// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t M_start_block = get_compile_time_arg_val(0);
    constexpr uint32_t M_end_block = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_start_block = get_compile_time_arg_val(3);
    constexpr uint32_t N_end_block = get_compile_time_arg_val(4);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t input_tile_size = get_compile_time_arg_val(8);
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(9));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(11);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(argidx++);

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<12>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, input_tile_size);

    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    const uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x,
        in0_mcast_dest_noc_start_y,
        in0_mcast_dest_noc_end_x,
        in0_mcast_dest_noc_end_y,
        in0_mcast_receiver_semaphore_addr);

    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y, in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y, 0);

    for (uint32_t m_block = M_start_block; m_block <= M_end_block; m_block++) {
        for (uint32_t n_block = N_start_block; n_block <= N_end_block; n_block++) {
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                DPRINT << "read in0 on m_block: " << m_block << ", n_block: " << n_block << ", k_block: " << k_block
                       << ENDL();
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);

#ifndef SKIP_IN0
                uint32_t in0_write_ptr = get_write_ptr(cb_id_in0);
                uint32_t in0_start_address = in0_write_ptr;

                for (uint32_t m = 0; m < M_block_tiles; m++) {
                    uint32_t m_id = m_block * M_block_tiles + m;
                    for (uint32_t k = 0; k < K_block_tiles; k++) {
                        uint32_t k_id = k_block * K_block_tiles + k;
                        uint32_t tile_id = m_id * K_tiles + k_id;
                        DPRINT << "read in0 tile " << tile_id << ENDL();
                        noc_async_read_tile(tile_id, in0_reader, in0_write_ptr);
                        in0_write_ptr += input_tile_size;
                    }
                }
                noc_async_read_barrier();

                DPRINT << "in0 sender wait for all clear" << ENDL();
                noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
                noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                uint64_t in0_multicast_data_addr = in0_multicast_data_noc | in0_start_address;

                DPRINT << "in0 sender before send" << ENDL();
                noc_async_write_multicast(
                    in0_start_address,
                    in0_multicast_data_addr,
                    in0_block_num_tiles * input_tile_size,
                    in0_mcast_num_dests,
                    true);
                DPRINT << "in0 sender after send" << ENDL();

                noc_semaphore_set_multicast(
                    in0_mcast_receiver_semaphore_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_dests);
                DPRINT << "in0 sender after send data arrived" << ENDL();
#endif
                cb_push_back(cb_id_in0, in0_block_num_tiles);
            }
        }
    }
}
