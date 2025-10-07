// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/dprint_pages.h"

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
    constexpr uint32_t buffer_factor = get_compile_time_arg_val(10);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_mcast_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_mcast_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    uint32_t* in1_valid_sem_ids = reinterpret_cast<uint32_t*>(get_arg_addr(argidx));
    argidx += buffer_factor;
    uint32_t* in1_ack_sem_ids = reinterpret_cast<uint32_t*>(get_arg_addr(argidx));

    uint32_t in1_valid_sem_addrs[buffer_factor];
    uint32_t in1_ack_sem_addrs[buffer_factor];
    for (uint32_t i = 0; i < buffer_factor; i++) {
        in1_valid_sem_addrs[i] = get_semaphore(in1_valid_sem_ids[i]);
        in1_ack_sem_addrs[i] = get_semaphore(in1_ack_sem_ids[i]);
    }

    // Tensor accessor for output tensor
    constexpr auto out_args = TensorAccessorArgs<11>();
    const auto out_reader = TensorAccessor(out_args, out_addr, input_tile_size);

    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in1_dm_out = tt::CBIndex::c_3;

    const uint64_t in1_sender_base_noc_addr = get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, 0);

    DPRINT << "in1recv: M_start_block: " << M_start_block << ", M_end_block: " << M_end_block
           << ", N_start_block: " << N_start_block << ", N_end_block: " << N_end_block << ENDL();

    constexpr uint32_t N_num_blocks = N_end_block - N_start_block + 1;

    bool k_forward = true;
    bool n_forward = true;
    bool reuse_block = false;

    uint32_t num_received = 0;
    uint32_t buf_idx = 0;
    for (uint32_t m_block = M_start_block; m_block <= M_end_block; m_block++) {
        for (uint32_t n_block_iter = 0; n_block_iter < N_num_blocks; n_block_iter++) {
            uint32_t n_block = n_forward ? N_start_block + n_block_iter : N_end_block - n_block_iter;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                DPRINT << "in1recv: read in1 on m_block: " << m_block << ", n_block: " << n_block
                       << ", k_block: " << k_block << ENDL();
                if (reuse_block && k_block_iter == 0) {
                    reuse_block = false;
                    continue;
                }
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

#ifndef SKIP_IN1
                volatile tt_l1_ptr uint32_t* in1_valid_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_valid_sem_addrs[buf_idx]);
                uint64_t in1_ack_sem_noc_addr = in1_sender_base_noc_addr | in1_ack_sem_addrs[buf_idx];
                if (num_received >= buffer_factor) {
                    noc_semaphore_inc(in1_ack_sem_noc_addr, 1);
                } else {
                    // Only increment while the buffer has not filled at least once to avoid overflow
                    num_received++;
                }
                noc_semaphore_wait(in1_valid_sem_ptr, VALID);
                noc_semaphore_set(in1_valid_sem_ptr, INVALID);
                buf_idx = (buf_idx + 1) % buffer_factor;
#endif

                cb_push_back(cb_id_in1, in1_block_num_tiles);
            }
            k_forward = !k_forward;
            // We have an output block to write out
            cb_wait_front(cb_id_in1_dm_out, out_block_num_tiles);
            if (n_block_iter == (N_num_blocks - 1)) {
                // This is the last iteration of the N block loop, so we will stride M next and get reuse, so we should
                // write the output.

#ifndef SKIP_OUT
                // uint32_t out_read_ptr = get_read_ptr(cb_id_in1_dm_out);
                // // safe_print_bf16_tile(out_read_ptr);
                // DPRINT << "in1recv: write out on m_block: " << m_block << ", n_block: " << n_block << ENDL();
                // for (uint32_t m = 0; m < M_block_tiles; m++) {
                //     uint32_t m_id = m_block * M_block_tiles + m;
                //     for (uint32_t n = 0; n < N_block_tiles; n++) {
                //         uint32_t n_id = n_block * N_block_tiles + n;
                //         uint32_t tile_id = m_id * N_tiles + n_id;
                //         // DPRINT << "write out tile " << tile_id << ENDL();
                //         noc_async_write_tile(tile_id, out_reader, out_read_ptr);
                //         out_read_ptr += input_tile_size;
                //     }
                // }
                // noc_async_writes_flushed();
#endif
            }
            cb_pop_front(cb_id_in1_dm_out, out_block_num_tiles);
        }
        n_forward = !n_forward;
        // We get reuse on in1 when striding M block
        reuse_block = true;
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
