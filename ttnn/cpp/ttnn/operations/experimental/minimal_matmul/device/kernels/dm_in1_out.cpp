// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/dprint_pages.h"

void safe_print_bf16_tile(uint32_t cb_id) {
#if defined(DEBUG_PRINT_ENABLED)
    tt::data_movement::common::print_bf16_tile(cb_id);
#endif
}

void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t input_tile_size = get_compile_time_arg_val(6);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);

    // Tensor accessor for input tensor
    constexpr auto in1_args = TensorAccessorArgs<7>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr, input_tile_size);
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto out_reader = TensorAccessor(out_args, out_addr, input_tile_size);

    constexpr uint32_t M_num_blocks = M_tiles / M_block_tiles;
    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t N_num_blocks = N_tiles / N_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;

    for (uint32_t m_block = 0; m_block < M_num_blocks; m_block++) {
        for (uint32_t n_block = 0; n_block < N_num_blocks; n_block++) {
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                DPRINT << "read in1 on m_block: " << m_block << ", n_block: " << n_block << ", k_block: " << k_block
                       << ENDL();
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

#ifndef SKIP_IN1
                uint32_t in1_write_ptr = get_write_ptr(cb_id_in1);
                for (uint32_t k = 0; k < K_block_tiles; k++) {
                    uint32_t k_id = k_block * K_block_tiles + k;
                    for (uint32_t n = 0; n < N_block_tiles; n++) {
                        uint32_t n_id = n_block * N_block_tiles + n;
                        uint32_t tile_id = k_id * N_tiles + n_id;
                        DPRINT << "read in1 tile " << tile_id << ENDL();
                        noc_async_read_tile(tile_id, in1_reader, in1_write_ptr);
                        in1_write_ptr += input_tile_size;
                    }
                }
                noc_async_read_barrier();
#endif

                cb_push_back(cb_id_in1, in1_block_num_tiles);
            }
            // We have an output block to write out
            cb_wait_front(cb_id_out, out_block_num_tiles);

#ifndef SKIP_OUT
            uint32_t out_read_ptr = get_read_ptr(cb_id_out);
            // safe_print_bf16_tile(out_read_ptr);
            DPRINT << "write out on m_block: " << m_block << ", n_block: " << n_block << ENDL();
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                uint32_t m_id = m_block * M_block_tiles + m;
                for (uint32_t n = 0; n < N_block_tiles; n++) {
                    uint32_t n_id = n_block * N_block_tiles + n;
                    uint32_t tile_id = m_id * N_tiles + n_id;
                    DPRINT << "write out tile " << tile_id << ENDL();
                    noc_async_write_tile(tile_id, out_reader, out_read_ptr);
                    out_read_ptr += input_tile_size;
                }
            }
            noc_async_write_barrier();
#endif

            cb_pop_front(cb_id_out, out_block_num_tiles);
        }
    }
}
