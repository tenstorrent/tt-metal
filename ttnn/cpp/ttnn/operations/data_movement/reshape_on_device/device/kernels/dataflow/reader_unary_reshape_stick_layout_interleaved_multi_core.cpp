// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_stick_size = get_compile_time_arg_val(1);

    constexpr auto cb_in0 = tt::CB::c_in0;

#define stick_size_is_pow2 get_compile_time_arg_val(2) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(3);
#else
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
#endif
#if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<src_is_dram> s = {.bank_base_address = src_addr,
                                                   .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src_is_dram> s = {.bank_base_address = src_addr, .page_size = page_size};
#endif

    uint32_t i_stick = start_id;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_reserve_back(cb_in0, num_sticks_per_cb_push);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            uint64_t read_noc_addr = get_noc_addr(i_stick, s);
            noc_async_read(read_noc_addr, l1_write_addr, old_stick_size);
            l1_write_addr += old_stick_size;
            i_stick++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, num_sticks_per_cb_push);
    }
}
