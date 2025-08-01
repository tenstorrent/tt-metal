// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr auto input_args = TensorAccessorArgs<2>();

    const auto s = TensorAccessor(input_args, input_buffer_address, input_page_size);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(src_cb_id, 1);
        uint64_t input_noc_addr = get_noc_addr(i, s);
        uint32_t src_cb_write_addr = get_write_ptr(src_cb_id);
        noc_async_read(input_noc_addr, src_cb_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(src_cb_id, 1);
    }
}
