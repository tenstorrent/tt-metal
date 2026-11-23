// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-reader (sender core, NCRISC / NoC0): read x one round at a time
// (block_m_tiles x block_k_tiles tiles, (m, k) row-major) into cb_x_resident, one barrier and one
// push per round. The multicast kernel on BRISC / NoC1 consumes each round, so the DRAM reads of
// round r+1 overlap the multicast of round r on the other NoC.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);

    for (uint32_t r = 0; r < num_k_blocks; ++r) {
        cb_reserve_back(cb_x_resident, round_tiles);
        uint32_t dst = get_write_ptr(cb_x_resident);
        for (uint32_t m = 0; m < block_m_tiles; ++m) {
            const uint32_t row_page = m * tensor_k_tiles + r * block_k_tiles;
            for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
                noc_async_read_tile(row_page + kk, x_acc, dst);
                dst += x_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x_resident, round_tiles);
    }
}
