// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_stride = get_compile_time_arg_val(2);
    constexpr uint32_t num_input_tensors = get_compile_time_arg_val(3);

    const uint32_t base_l1_write_addr = get_write_ptr(output_cb);

    uint32_t arg_idx = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const uint32_t input_num_pages_per_stick = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_num_sticks = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_write_offset = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_read_offset = get_arg_val<uint32_t>(arg_idx++);

        uint32_t l1_write_addr = base_l1_write_addr + input_write_offset;
        uint32_t l1_read_addr = get_read_ptr(input_id) + input_read_offset;
        noc_async_read_one_packet_set_state(get_noc_addr(l1_read_addr), page_size);

        for (uint32_t stick_idx = 0; stick_idx < input_num_sticks; stick_idx++) {
            for (uint32_t page_idx = 0; page_idx < input_num_pages_per_stick; page_idx++) {
                noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr + page_size * page_idx);
                l1_read_addr += page_size;
            }
            l1_write_addr += output_stride;
        }
    }

    noc_async_read_barrier();
}
