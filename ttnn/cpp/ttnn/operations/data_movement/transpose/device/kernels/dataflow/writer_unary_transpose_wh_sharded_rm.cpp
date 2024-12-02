// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_hw_blocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t W_per_tile = get_compile_time_arg_val(3);
    constexpr uint32_t W_per_tile_last = get_compile_time_arg_val(4);
    constexpr uint32_t H_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t l1_read_offset_bytes = get_compile_time_arg_val(6);

    constexpr auto cb_out = tt::CBIndex::c_27;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    const uint32_t stick_size_bytes = H_size_bytes;

    uint32_t l1_write_addr = get_write_ptr(cb_out0);
    uint64_t write_noc_addr = get_noc_addr(l1_write_addr);

    // temporary fix until pack_untilze is fully fixed
    if constexpr (Ht > 8) {
        noc_async_write_one_packet_set_state(write_noc_addr, stick_size_bytes);

        for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
            for (uint32_t w = 0; w < Wt; ++w) {
                cb_wait_front(cb_out, Ht);
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                uint32_t W_curr = w == Wt - 1 ? W_per_tile_last : W_per_tile;
                for (uint32_t w_datum = 0; w_datum < W_curr; ++w_datum) {
                    noc_async_write_one_packet_with_state(l1_read_addr, write_noc_addr);
                    l1_read_addr += l1_read_offset_bytes;
                    write_noc_addr += stick_size_bytes;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, Ht);
            }
        }
        noc_async_write_barrier();
    }
}
