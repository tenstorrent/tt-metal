// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_padded_aligned = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto s = TensorAccessor(dst_args, dst_addr, stick_size_bytes);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_wait_front(cb_out0, num_sticks_per_barrier);

        uint32_t l1_read_addr = get_read_ptr(cb_out0);

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            uint64_t write_noc_addr = get_noc_addr(i_stick, s);
            noc_async_write(l1_read_addr, write_noc_addr, stick_size_bytes);
            l1_read_addr += stick_size_padded_aligned;
            i_stick += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out0, num_sticks_per_barrier);
    }
}
