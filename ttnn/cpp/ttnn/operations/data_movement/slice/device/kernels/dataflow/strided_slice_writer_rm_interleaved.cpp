// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {




    constexpr bool dst0_is_dram = (bool) get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);

    const InterleavedAddrGen<dst0_is_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = page_size
    };

    constexpr uint32_t cb_id_out0 = 24;
    const uint32_t start_id = 0;
    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;

    for (uint32_t iter = i_stick; iter < num_sticks_per_core; ++iter) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        uint64_t dst_noc_addr = get_noc_addr(iter, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
