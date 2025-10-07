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
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t in1_mcast_num_dests = get_compile_time_arg_val(12);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(argidx++);

    // Tensor accessor for input tensor
    constexpr auto in1_args = TensorAccessorArgs<13>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr, input_tile_size);
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto out_reader = TensorAccessor(out_args, out_addr, input_tile_size);

    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_in1_dm_out = tt::CBIndex::c_3;

    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);
    *(in1_mcast_receiver_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in1_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_sender_semaphore_addr);

    const uint64_t in1_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        in1_mcast_receiver_semaphore_addr);

    const uint64_t in1_multicast_data_noc = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x, in1_mcast_dest_noc_start_y, in1_mcast_dest_noc_end_x, in1_mcast_dest_noc_end_y, 0);

    DPRINT << "in1send: M_start_block: " << M_start_block << ", M_end_block: " << M_end_block
           << ", N_start_block: " << N_start_block << ", N_end_block: " << N_end_block << ENDL();

    constexpr uint32_t N_num_blocks = N_end_block - N_start_block + 1;
    bool k_forward = true;
    bool n_forward = true;
    bool reuse_block = false;
    for (uint32_t m_block = M_start_block; m_block <= M_end_block; m_block++) {
        for (uint32_t n_block_iter = 0; n_block_iter < N_num_blocks; n_block_iter++) {
            uint32_t n_block = n_forward ? N_start_block + n_block_iter : N_end_block - n_block_iter;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                DPRINT << "in1send: read in1 on m_block: " << m_block << ", n_block: " << n_block
                       << ", k_block: " << k_block << ENDL();
                if (reuse_block && k_block_iter == 0) {
                    reuse_block = false;
                    continue;
                }
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

#ifndef SKIP_IN1
                uint32_t in1_write_ptr = get_write_ptr(cb_id_in1);
                uint32_t in1_start_address = in1_write_ptr;
                for (uint32_t k = 0; k < K_block_tiles; k++) {
                    uint32_t k_id = k_block * K_block_tiles + k;
                    for (uint32_t n = 0; n < N_block_tiles; n++) {
                        uint32_t n_id = n_block * N_block_tiles + n;
                        uint32_t tile_id = k_id * N_tiles + n_id;
                        // DPRINT << "read in1 tile " << tile_id << ENDL();
                        noc_async_read_tile(tile_id, in1_reader, in1_write_ptr);
                        in1_write_ptr += input_tile_size;
                    }
                }
                noc_async_read_barrier();
#endif
                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in1, in1_block_num_tiles);
#ifndef SKIP_IN1

                noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
                noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

                uint64_t in1_multicast_data_addr = in1_multicast_data_noc | in1_start_address;

                noc_async_write_multicast(
                    in1_start_address,
                    in1_multicast_data_addr,
                    in1_block_num_tiles * input_tile_size,
                    in1_mcast_num_dests,
                    true);

                noc_semaphore_set_multicast(
                    in1_mcast_receiver_semaphore_addr, in1_mcast_receiver_semaphore_noc_addr, in1_mcast_num_dests);
#endif
            }
            k_forward = !k_forward;
            // We have an output block to write out
            cb_wait_front(cb_id_in1_dm_out, out_block_num_tiles);
            if (n_block_iter == (N_num_blocks - 1)) {
                // This is the last iteration of the N block loop, so we will stride M next and get reuse, so we should
                // write the output.

#ifndef SKIP_OUT
                uint32_t out_read_ptr = get_read_ptr(cb_id_in1_dm_out);
                // safe_print_bf16_tile(out_read_ptr);
                DPRINT << "in1recv: write out on m_block: " << m_block << ", n_block: " << n_block << ENDL();
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
                noc_async_writes_flushed();
#endif
            }
            cb_pop_front(cb_id_in1_dm_out, out_block_num_tiles);
        }
        n_forward = !n_forward;
        // We get reuse on in1 when striding M block
        reuse_block = true;
    }
}
