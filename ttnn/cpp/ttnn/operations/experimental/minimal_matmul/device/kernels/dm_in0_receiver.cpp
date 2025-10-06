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

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);

    const uint64_t in0_mcast_sender_semaphore_noc_addr =
        get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);

    DPRINT << "in0recv: M_start_block: " << M_start_block << ", M_end_block: " << M_end_block
           << ", N_start_block: " << N_start_block << ", N_end_block: " << N_end_block << ENDL();

    constexpr uint32_t N_num_blocks = N_end_block - N_start_block + 1;

    bool k_forward = true;
    bool n_forward = true;
    bool reuse_block = false;
    for (uint32_t m_block = M_start_block; m_block <= M_end_block; m_block++) {
        reuse_block = false;
        for (uint32_t n_block_iter = 0; n_block_iter < N_num_blocks; n_block_iter++) {
            uint32_t n_block = n_forward ? N_start_block + n_block_iter : N_end_block - n_block_iter;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                DPRINT << "in0recv: read in0 on m_block: " << m_block << ", n_block: " << n_block
                       << ", k_block: " << k_block << ENDL();
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);

#ifndef SKIP_IN0
                noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);
                noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);
                noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);
#endif

                cb_push_back(cb_id_in0, in0_block_num_tiles);
            }
            k_forward = !k_forward;
            // We get reuse on in0 when striding N block
            reuse_block = true;
        }
        n_forward = !n_forward;
    }
}
