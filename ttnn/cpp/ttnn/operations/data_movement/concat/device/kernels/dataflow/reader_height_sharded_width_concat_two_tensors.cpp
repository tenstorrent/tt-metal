
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);

    constexpr uint32_t input_stick_size_0 = get_compile_time_arg_val(1);
    constexpr uint32_t input_stick_size_1 = get_compile_time_arg_val(2);
    constexpr uint32_t input_stride_0 = get_compile_time_arg_val(3);
    constexpr uint32_t input_stride_1 = get_compile_time_arg_val(4);

    const uint32_t num_output_pages = get_compile_time_arg_val(5);
    const uint32_t page_start = get_compile_time_arg_val(6);
    const uint32_t page_end = get_compile_time_arg_val(7);
    const uint32_t output_stick_offset = get_compile_time_arg_val(8);
    const uint32_t input_start_0 = get_compile_time_arg_val(9);
    const uint32_t input_start_1 = get_compile_time_arg_val(10);

    const uint32_t groups = get_compile_time_arg_val(11);

    constexpr uint32_t group_stick_size_0 = input_stick_size_0 / groups;
    constexpr uint32_t group_stick_size_1 = input_stick_size_1 / groups;
    constexpr uint32_t group_stride_0 = input_stride_0 / groups;
    constexpr uint32_t group_stride_1 = input_stride_1 / groups;

    const uint32_t base_l1_write_addr = get_write_ptr(output_cb);

    uint32_t l1_write_addr_0 = base_l1_write_addr + output_stick_offset;
    const uint32_t l1_read_addr_0 = get_read_ptr(0) + input_start_0;
    const uint64_t noc_addr_0 = get_noc_addr(l1_read_addr_0);
    noc_async_read_one_packet_set_state(noc_addr_0, group_stick_size_0);

    uint32_t read_offset_0 = l1_read_addr_0;
    uint32_t l1_write_addr_inc_0 = group_stick_size_0 + group_stride_0;
    for (uint32_t page_id_input = page_start; page_id_input < page_end; page_id_input++) {
        for (uint32_t i = 0; i < groups; i++) {
            noc_async_read_one_packet_with_state<true>(read_offset_0, l1_write_addr_0);
            l1_write_addr_0 += (l1_write_addr_inc_0);
            read_offset_0 += group_stick_size_0;
        }
    }

    uint32_t l1_write_addr_1 = base_l1_write_addr + output_stick_offset + group_stick_size_0;
    const uint32_t l1_read_addr_1 = get_read_ptr(1) + input_start_1;
    const uint64_t noc_addr_1 = get_noc_addr(l1_read_addr_1);
    noc_async_read_one_packet_set_state(noc_addr_1, group_stick_size_1);

    uint32_t read_offset_1 = l1_read_addr_1;
    uint32_t l1_write_addr_inc_1 = group_stick_size_1 + group_stride_1;
    for (uint32_t page_id_input = page_start; page_id_input < page_end; page_id_input++) {
        for (uint32_t i = 0; i < groups; i++) {
            noc_async_read_one_packet_with_state<true>(read_offset_1, l1_write_addr_1);
            l1_write_addr_1 += (l1_write_addr_inc_1);
            read_offset_1 += group_stick_size_1;
        }
    }

    noc_async_read_barrier();
}
