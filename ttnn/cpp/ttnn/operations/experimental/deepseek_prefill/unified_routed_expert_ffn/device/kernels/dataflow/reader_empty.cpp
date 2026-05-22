// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DIAGNOSTIC: reader does phase 1 reads (gate matmul) but skips phase 2/4.
// Compute pops what reader pushes and exits.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_mt = get_arg_val<uint32_t>(8);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(9);
    const uint32_t chunk_start_tile_row = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_in0_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1_gate = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(9);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(13);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(15);

    constexpr uint32_t g_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;

    constexpr uint32_t x_accessor_offset = 19;
    constexpr auto x_args = TensorAccessorArgs<x_accessor_offset>();
    const auto x_acc = TensorAccessor(x_args, x_addr, get_tile_size(cb_in0_x));

    constexpr uint32_t gate_accessor_offset = x_args.next_compile_time_args_offset();
    constexpr auto gate_args = TensorAccessorArgs<gate_accessor_offset>();
    const auto gate_acc = TensorAccessor(gate_args, gate_addr, get_tile_size(cb_in1_gate));

    const uint32_t this_core_first_row = chunk_start_tile_row + my_mt * per_core_M;
    const uint32_t x_tile_bytes = get_tile_size(cb_in0_x);
    const uint32_t gate_tile_bytes = get_tile_size(cb_in1_gate);

    for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
        cb_reserve_back(cb_in0_x, g_in0_block_num_tiles);
        uint32_t l1_x = get_write_ptr(cb_in0_x);
        for (uint32_t m = 0; m < per_core_M; ++m) {
            for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                const uint32_t row = this_core_first_row + m;
                const uint32_t col = kb * in0_block_w_gu + k;
                const uint32_t tile_idx = row * K_gate_tiles + col;
                noc_async_read_tile(tile_idx, x_acc, l1_x);
                l1_x += x_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0_x, g_in0_block_num_tiles);

        cb_reserve_back(cb_in1_gate, g_in1_block_num_tiles);
        uint32_t l1_w = get_write_ptr(cb_in1_gate);
        for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
            for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                const uint32_t row = kb * in0_block_w_gu + k;
                const uint32_t col = my_nt_gu * per_core_N_gu + n;
                const uint32_t tile_idx = row * N_gate_tiles_full + col;
                noc_async_read_tile(tile_idx, gate_acc, l1_w);
                l1_w += gate_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in1_gate, g_in1_block_num_tiles);
    }
}
