// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows = get_compile_time_arg_val(3);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = page_size};

    uint32_t curr_addr = src_addr;
    for (uint32_t i = 0; i < num_rows; ++i) {
        cb_reserve_back(tt::CBIndex::c_0, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
        noc_async_read_page(i, s0, src_buffer_l1_addr);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* out_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_buffer_l1_addr);
        cb_push_back(tt::CBIndex::c_0, 1);
    }
}
