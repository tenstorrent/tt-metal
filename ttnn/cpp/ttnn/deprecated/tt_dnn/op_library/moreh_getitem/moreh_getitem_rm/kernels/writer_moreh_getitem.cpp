// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    // buffer
    uint32_t dst_addr = get_arg_val<uint32_t>(i++);

    // output
    uint32_t output_stick_size = get_arg_val<uint32_t>(i++);

    // etc
    uint32_t start_id = get_arg_val<uint32_t>(i++);
    uint32_t num_sticks = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_out = tt::CB::c_in0;

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGen<dst_is_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = output_stick_size
    };

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++ i) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint64_t dst_noc_addr = get_noc_addr(i, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, output_stick_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }
}
