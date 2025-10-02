// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

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
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<6>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, input_tile_size);

    constexpr uint32_t M_num_blocks = M_tiles / M_block_tiles;
    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t N_num_blocks = N_tiles / N_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    for (uint32_t m_block = 0; m_block < M_num_blocks; m_block++) {
        for (uint32_t n_block = 0; n_block < N_num_blocks; n_block++) {
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);
                uint32_t in0_write_ptr = get_write_ptr(cb_id_in0);
                for (uint32_t m = 0; m < M_block_tiles; m++) {
                    uint32_t m_id = m_block * M_block_tiles + m;
                    for (uint32_t k = 0; k < K_block_tiles; k++) {
                        uint32_t k_id = k_block * K_block_tiles + k;
                        uint32_t tile_id = m_id * K_tiles + k_id;
                        noc_async_read_tile(tile_id, in0_reader, in0_write_ptr);
                        in0_write_ptr += input_tile_size;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, in0_block_num_tiles);
            }
        }
    }
}
