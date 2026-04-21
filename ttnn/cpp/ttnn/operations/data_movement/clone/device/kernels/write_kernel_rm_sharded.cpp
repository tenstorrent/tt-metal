// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    experimental::CircularBuffer dst_cb(dst_cb_id);

    uint64_t local_l1_write_addr = get_noc_addr(output_buffer_address);

    for (uint32_t i = 0; i < num_sticks; ++i) {
        dst_cb.wait_front(1);
        uint32_t dst_cb_read_addr = dst_cb.get_read_ptr();

        noc_async_write(dst_cb_read_addr, local_l1_write_addr, stick_size);
        noc_async_write_barrier();

        dst_cb.pop_front(1);
        local_l1_write_addr += stick_size;
    }
}
