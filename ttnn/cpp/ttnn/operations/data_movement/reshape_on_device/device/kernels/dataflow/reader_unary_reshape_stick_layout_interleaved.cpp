// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t cb_id_in0 = 0;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t stick_size = get_arg_val<uint32_t>(2);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_c = stick_size / 64;  // Assuming 2 bytes per datum, there are 64 bytes per tile row
    uint32_t stick_id = 0;

    constexpr bool stick_size_is_power_of_two = (get_compile_time_arg_val(1) == 1);
#if (stick_size_is_power_of_two)
    const uint32_t log_base_2_of_page_size = get_arg_val<uint32_t>(3);
    const InterleavedPow2AddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr,

        .log_base_2_of_page_size = log_base_2_of_page_size  // TODO(AP): refactor
    };
#else
    const InterleavedAddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr,

        .page_size = stick_size};
#endif

    for (uint32_t i = 0; i < num_sticks / 32; i++) {
        // We reserve back an entire tile row and issue a bunch of reads
        cb_reserve_back(cb_id_in0, num_tiles_c);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < 32; j++) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s);

            noc_async_read(src_noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_tiles_c);
    }
}
