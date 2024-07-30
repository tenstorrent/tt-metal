// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"


void kernel_main() {

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_id  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(2);

    const uint32_t stick_size_bytes = W_size_bytes;

    #define stick_size_is_pow2 get_compile_time_arg_val(3) == 1
    #if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(4);
    #else
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    #endif
    #if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size
    };
    #else
    const InterleavedAddrGen<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = page_size
    };
    #endif

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_wait_front(cb_out0, num_read_per_barrier);

        uint32_t l1_read_addr = get_read_ptr(cb_out0);

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            uint64_t write_noc_addr = get_noc_addr(i_stick, s);
            noc_async_write(l1_read_addr, write_noc_addr, stick_size_bytes);
            l1_read_addr += stick_size_bytes;
            i_stick += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out0, num_read_per_barrier);
    }

}
