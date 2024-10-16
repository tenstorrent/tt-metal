// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_hw_blocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t H_per_tile = get_compile_time_arg_val(2);
    constexpr uint32_t H_per_tile_last = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t l1_write_offset_bytes = get_compile_time_arg_val(6);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in = tt::CB::c_intermed0;

    const uint32_t stick_size_bytes = W_size_bytes;

    uint32_t l1_read_addr = get_read_ptr(cb_in0);
    uint64_t read_noc_addr = get_noc_addr(l1_read_addr);

    noc_async_read_one_packet_set_state(read_noc_addr, stick_size_bytes);

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_reserve_back(cb_in, Wt);
            uint32_t l1_write_addr = get_write_ptr(cb_in);
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                noc_async_read_one_packet_with_state(read_noc_addr, l1_write_addr);
                l1_write_addr += l1_write_offset_bytes;
                read_noc_addr += stick_size_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, Wt);
        }
    }
}
