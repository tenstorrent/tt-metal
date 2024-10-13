// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr bool output_is_dram = get_compile_time_arg_val(1) == 1;

#define dst_stick_size_is_power_of_two get_compile_time_arg_val(2) == 1
#if (dst_stick_size_is_power_of_two)
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);
    const InterleavedPow2AddrGen<output_is_dram> s = {
        .bank_base_address = get_arg_val<uint32_t>(0),
        .log_base_2_of_page_size = dst_log_base_2_of_page_size,
    };
#else
    const InterleavedAddrGen<output_is_dram> s = {
        .bank_base_address = get_arg_val<uint32_t>(0),
        .page_size = stick_size,
    };
#endif

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(dst_cb_id, 1);
        noc_async_write(get_read_ptr(dst_cb_id), get_noc_addr(i, s), stick_size);
        noc_async_write_barrier();
        cb_pop_front(dst_cb_id, 1);
    }
}
