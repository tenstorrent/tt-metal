// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t new_stick_size = get_compile_time_arg_val(2);

    constexpr bool stick_size_is_pow2 = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t size = get_compile_time_arg_val(4);

    const auto s = get_interleaved_addr_gen<dst_is_dram, stick_size_is_pow2>(dst_addr, size);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_wait_front(cb_out0, num_sticks_per_cb_push);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            uint64_t write_noc_addr = get_noc_addr(i_stick, s);
            noc_async_write(l1_read_addr, write_noc_addr, new_stick_size);
            l1_read_addr += new_stick_size;
            i_stick += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out0, num_sticks_per_cb_push);
    }
}
