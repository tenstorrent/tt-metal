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
    constexpr uint32_t N_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_start_block = get_compile_time_arg_val(4);
    constexpr uint32_t N_end_block = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t input_tile_size = get_compile_time_arg_val(9);
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr auto out_args = TensorAccessorArgs<12>();
    const auto out_reader = TensorAccessor(out_args, out_addr, input_tile_size);
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;

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
            if (n_block_iter < N_num_blocks - 1) {
                // This is not the last iteration of the N block loop, so we will get reuse on next block, so we should
                // write output
                cb_wait_front(cb_id_out, out_block_num_tiles);

#ifndef SKIP_OUT
                uint32_t out_read_ptr = get_read_ptr(cb_id_out);
                // safe_print_bf16_tile(out_read_ptr);
                DPRINT << "in0send: write out on m_block: " << m_block << ", n_block: " << n_block << ENDL();
                for (uint32_t m = 0; m < M_block_tiles; m++) {
                    uint32_t m_id = m_block * M_block_tiles + m;
                    for (uint32_t n = 0; n < N_block_tiles; n++) {
                        uint32_t n_id = n_block * N_block_tiles + n;
                        uint32_t tile_id = m_id * N_tiles + n_id;
                        // DPRINT << "write out tile " << tile_id << ENDL();
                        noc_async_write_tile(tile_id, out_reader, out_read_ptr);
                        out_read_ptr += input_tile_size;
                    }
                }
                noc_async_write_barrier();
#endif

                cb_pop_front(cb_id_out, out_block_num_tiles);
            }
        }
        n_forward = !n_forward;
    }
}
