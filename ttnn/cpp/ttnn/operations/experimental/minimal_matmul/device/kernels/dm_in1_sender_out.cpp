// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "matmul_dataflow_common.hpp"
#include "common.hpp"

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
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(9);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(11);
    uint32_t in1_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(12));
    uint32_t in1_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(13));
    uint32_t in1_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(15);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(16);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);

    // Tensor accessor for input tensor
    constexpr auto in1_args = TensorAccessorArgs<17>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr, in1_tile_size);
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto out_reader = TensorAccessor(out_args, out_addr, out_tile_size);
#ifdef FUSE_BIAS
    constexpr auto in2_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr, in2_tile_size);
#endif

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_4;
#endif

    volatile tt_l1_ptr uint32_t* in1_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_valid_semaphore_addr);
    *(in1_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in1_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in1_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_semaphore_addr);
    const uint64_t in1_sender_semaphore_noc_addr =
        get_noc_addr(in1_sender_noc_x, in1_sender_noc_y, in1_sender_semaphore_addr);

    const uint64_t in1_receiver_semaphore_noc_addr =
        get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, in1_receiver_semaphore_addr);

    const uint32_t M_tiles_per_core = M_end_tile - M_start_tile;
    const uint32_t N_tiles_per_core = N_end_tile - N_start_tile;
    const uint32_t N_num_blocks = div_up(N_tiles_per_core, N_block_tiles);
    const uint32_t M_num_blocks = div_up(M_tiles_per_core, M_block_tiles);

    bool k_forward = true;
    bool n_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_n_tile = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_num_blocks; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        for (uint32_t n_block_iter = 0; n_block_iter < N_num_blocks; n_block_iter++) {
            uint32_t n_tile = n_forward ? N_start_tile + n_block_iter * N_block_tiles
                                        : N_start_tile + (N_num_blocks - 1 - n_block_iter) * N_block_tiles;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        cb_wait_front(cb_id_out, out_block_num_tiles);
                        uint32_t out_read_ptr = get_read_ptr(cb_id_out);
                        write_block_sync(
                            out_reader,
                            out_shape,
                            out_read_ptr,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile + M_block_tiles,
                            defer_write_n_tile,
                            defer_write_n_tile + N_block_tiles);
                        cb_pop_front(cb_id_out, out_block_num_tiles);
                    }
                }

                if (reuse_block && k_block_iter == 0) {
                    reuse_block = false;
                    continue;
                }
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

                uint32_t in1_start_address = get_write_ptr(cb_id_in1);
                if constexpr (is_injector_core) {
                    read_in1_block_sync(
                        in1_reader,
                        in1_shape,
                        in1_start_address,
                        in1_tile_size,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles,
                        n_tile,
                        n_tile + N_block_tiles);
                } else {
                    noc_semaphore_set(in1_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in1_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in1_receiver_semaphore_addr_ptr, VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in1, in1_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in1_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);

                    uint64_t in1_unicast_data_addr = get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, in1_start_address);

                    noc_async_write(in1_start_address, in1_unicast_data_addr, in1_block_num_tiles * in1_tile_size);

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                }
            }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_reserve_back(cb_id_in2, N_block_tiles);

                uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile + N_block_tiles; n_tile_id++) {
                    noc_async_read_tile(n_tile_id, in2_reader, l1_write_addr_in2);
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id_in2, N_block_tiles);
            }
#endif

            k_forward = !k_forward;
            // We have an output block to write out

            defer_write_m_tile = m_tile;
            defer_write_n_tile = n_tile;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !((m_block_iter == M_num_blocks - 1) && (n_block_iter == (N_num_blocks - 1)));
            defer_write = defer_write && !is_injector_core;

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    write_block_sync_granular(
                        out_reader,
                        out_shape,
                        cb_id_out,
                        N_block_tiles,
                        out_tile_size,
                        m_tile,
                        m_tile + M_block_tiles,
                        n_tile,
                        n_tile + N_block_tiles);
                }
            }
        }
        n_forward = !n_forward;
        // We get reuse on in1 when striding M block
        reuse_block = true;
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
