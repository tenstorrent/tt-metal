// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "matmul_dataflow_common.hpp"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t input_tile_size = get_compile_time_arg_val(9);
    uint32_t in0_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    uint32_t in0_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));
    uint32_t in0_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(13);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(14);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);

    DPRINT << "in0send: M_blocks: [" << M_start_block << ", " << M_end_block << "], N_blocks: [" << N_start_block
           << ", " << N_end_block << "]" << ENDL();

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<15>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, input_tile_size);
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out_reader = TensorAccessor(out_args, out_addr, input_tile_size);

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;

    volatile tt_l1_ptr uint32_t* in0_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_valid_semaphore_addr);
    *(in0_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);

    volatile tt_l1_ptr uint32_t* in0_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_semaphore_addr);
    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_semaphore_addr);

    const uint64_t in0_receiver_semaphore_noc_addr =
        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);

    /**
     * This is a Serpentine (Boustrophedon) output block ordering.
     * It enables reuse of one of the input blocks for the last output block.
     * Starting at output block (0,0), go east until the end, then south one block, then west until the end, then south
     * one block, and repeat. At the same time, alternate between K striding forwards or backwards in order to enable
     * reuse.
     */

    const uint32_t N_num_blocks = N_end_block - N_start_block + 1;

    bool k_forward = true;
    bool n_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_block = 0;
    uint32_t defer_write_n_block = 0;
    bool defer_write = false;

    for (uint32_t m_block = M_start_block; m_block <= M_end_block; m_block++) {
        reuse_block = false;
        for (uint32_t n_block_iter = 0; n_block_iter < N_num_blocks; n_block_iter++) {
            uint32_t n_block = n_forward ? N_start_block + n_block_iter : N_end_block - n_block_iter;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        DPRINT << "in0send: writing (M,N) = (" << m_block << ", " << n_block << ")" << ENDL();
                        write_block_sync_granular(
                            out_reader,
                            out_shape,
                            cb_id_out,
                            N_block_tiles,
                            input_tile_size,
                            defer_write_m_block * M_block_tiles,
                            (defer_write_m_block + 1) * M_block_tiles,
                            defer_write_n_block * N_block_tiles,
                            (defer_write_n_block + 1) * N_block_tiles);
                    }
                }

                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);

#ifndef SKIP_IN0
                uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                if constexpr (is_injector_core) {
                    read_in0_block_sync(
                        in0_reader,
                        in0_shape,
                        in0_start_address,
                        input_tile_size,
                        m_block * M_block_tiles,
                        (m_block + 1) * M_block_tiles,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                    // Get from previous device
                    DPRINT << "in0send: wait on (M,K) = (" << m_block << ", " << k_block << ") from previous device"
                           << ENDL();
                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
                    DPRINT << "in0send: wait done on (M,K) = (" << m_block << ", " << k_block
                           << ") from previous device" << ENDL();
                }
#endif
                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in0, in0_block_num_tiles);
#ifndef SKIP_IN0
                if (!is_sink_core) {
                    DPRINT << "in0send: forwarding (M,K) = (" << m_block << ", " << k_block << ") to next device"
                           << ENDL();

                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);

                    uint64_t in0_unicast_data_addr = get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);

                    noc_async_write(in0_start_address, in0_unicast_data_addr, in0_block_num_tiles * input_tile_size);

                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                    DPRINT << "in0send: forwarding done on (M,K) = (" << m_block << ", " << k_block
                           << ") to next device" << ENDL();
                }

#endif
            }
            k_forward = !k_forward;
            // We get reuse on in0 when striding N block
            reuse_block = true;

            defer_write_m_block = m_block;
            defer_write_n_block = n_block;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !((m_block == M_end_block) && (n_block_iter == (N_num_blocks - 1)));

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    DPRINT << "in0send: writing (M,N) = (" << m_block << ", " << n_block << ")" << ENDL();
                    write_block_sync_granular(
                        out_reader,
                        out_shape,
                        cb_id_out,
                        N_block_tiles,
                        input_tile_size,
                        m_block * M_block_tiles,
                        (m_block + 1) * M_block_tiles,
                        n_block * N_block_tiles,
                        (n_block + 1) * N_block_tiles);
                }
            }
        }
        n_forward = !n_forward;
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
