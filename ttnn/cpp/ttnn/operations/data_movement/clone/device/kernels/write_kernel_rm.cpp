// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr bool output_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr bool dst_stick_size_is_power_of_two = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);

    const auto s = get_interleaved_addr_gen<output_is_dram, dst_stick_size_is_power_of_two>(
        output_buffer_address, stick_size, dst_log_base_2_of_page_size);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(dst_cb_id, 1);
        uint32_t dst_cb_read_addr = get_read_ptr(dst_cb_id);
        uint64_t output_noc_addr = get_noc_addr(i, s);
        noc_async_write(dst_cb_read_addr, output_noc_addr, stick_size);
        noc_async_write_barrier();
        cb_pop_front(dst_cb_id, 1);
    }
}
