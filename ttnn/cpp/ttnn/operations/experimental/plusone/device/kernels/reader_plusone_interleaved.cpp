// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = stick_size};

    // Use cb as L1 scratch memory
    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (uint32_t h = 0; h < H; h++) {
        noc_async_read_page(h, s0, cb_addr);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < W; i++) {
            uint32_t val = stick[i];
            stick[i] = val + 1;
            // DPRINT << "val: " << val << ENDL();
        }

        uint64_t dst_noc_addr = get_noc_addr(h, s0);

        noc_async_write(cb_addr, dst_noc_addr, stick_size);
        noc_async_write_barrier();
    }
}
