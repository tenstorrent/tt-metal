// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);

    constexpr auto s0_args = TensorAccessorArgs<5>();
    const auto s0 = TensorAccessor(s0_args, src_addr, stick_size);

    // Use cb as L1 scratch memory
    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (uint32_t h = 0; h < H; h++) {
        if (src0_is_dram) {
            noc_async_read_page(h, s0, cb_addr);
            noc_async_read_barrier();
        }
        for (uint32_t i = 0; i < W; i++) {
            uint32_t val = stick[i];
            stick[i] = val + 1;
            // DPRINT << "val: " << val << ENDL();
        }
        if (src0_is_dram) {
            uint64_t dst_noc_addr = s0.get_noc_addr(h);
            noc_async_write(cb_addr, dst_noc_addr, stick_size);
            noc_async_write_barrier();
        }
    }
}
