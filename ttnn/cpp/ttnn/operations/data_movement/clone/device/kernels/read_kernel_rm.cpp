// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;

#define src_stick_size_is_power_of_two get_compile_time_arg_val(2) == 1
#if (src_stick_size_is_power_of_two)
    constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(3);
    const InterleavedPow2AddrGen<input_is_dram> s = {
        .bank_base_address = input_buffer_address,
        .log_base_2_of_page_size = src_log_base_2_of_page_size,
    };
#else
    const InterleavedAddrGen<input_is_dram> s = {
        .bank_base_address = input_buffer_address,
        .page_size = stick_size,
    };
#endif

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(src_cb_id, 1);
        uint64_t input_noc_addr = get_noc_addr(i, s);
        uint32_t src_cb_write_addr = get_write_ptr(src_cb_id);
        noc_async_read(input_noc_addr, src_cb_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(src_cb_id, 1);
    }
}
