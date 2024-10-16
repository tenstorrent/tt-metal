// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t num_tensors = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb = get_compile_time_arg_val(1);

    const uint32_t output_stick_offset = get_arg_val<uint32_t>(0);
    const uint32_t num_output_pages = get_arg_val<uint32_t>(1);
    const uint32_t page_start = get_arg_val<uint32_t>(2);
    const uint32_t page_end = get_arg_val<uint32_t>(3);
    uint32_t arg_index = 4;

    cb_reserve_back(output_cb, num_output_pages);
    const uint32_t base_l1_write_addr = get_write_ptr(output_cb);
    uint32_t output_offset = 0;
    for (uint32_t tensor_id = 0; tensor_id < num_tensors; tensor_id++) {
        const uint32_t input_stick_size = get_arg_val<uint32_t>(arg_index++);
        const uint32_t input_stride = get_arg_val<uint32_t>(arg_index++);
        const uint32_t input_start = get_arg_val<uint32_t>(arg_index++);
        const uint32_t l1_read_addr = get_read_ptr(tensor_id) + input_start;
        const uint64_t noc_addr = get_noc_addr(l1_read_addr);

        uint32_t l1_write_addr = base_l1_write_addr + output_stick_offset + output_offset;
        noc_async_read_one_packet_set_state(noc_addr, input_stick_size);
        uint32_t read_offset = l1_read_addr;
        for (uint32_t page_id_input = page_start; page_id_input < page_end; page_id_input++) {
            noc_async_read_one_packet_with_state<true>(read_offset, l1_write_addr);
            l1_write_addr += (input_stick_size + input_stride);
            read_offset += input_stick_size;
        }
        output_offset += input_stick_size;
    }
    noc_async_read_barrier();
    cb_push_back(output_cb, num_output_pages);
}
