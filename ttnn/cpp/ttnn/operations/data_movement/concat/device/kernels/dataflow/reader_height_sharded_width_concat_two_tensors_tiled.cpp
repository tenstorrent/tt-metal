// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <debug/dprint.h>

void kernel_main() {
    constexpr uint32_t input0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(3);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(4);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(5);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(6);
    constexpr uint32_t tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t groups = get_compile_time_arg_val(8);

    constexpr uint32_t input0_stride = tile_size * input0_num_tiles_width / groups;
    constexpr uint32_t input1_stride = tile_size * input1_num_tiles_width / groups;

    const uint32_t base_l1_read_addr_0 = get_read_ptr(input0_cb);
    const uint64_t noc_addr_0 = get_noc_addr(base_l1_read_addr_0);

    const uint32_t base_l1_read_addr_1 = get_read_ptr(input1_cb);
    const uint64_t noc_addr_1 = get_noc_addr(base_l1_read_addr_1);

    uint32_t base_l1_write_addr = get_write_ptr(output_cb);

    uint32_t l1_write_addr = base_l1_write_addr;
    uint32_t l1_read_addr = base_l1_read_addr_0;
    noc_async_read_one_packet_set_state(noc_addr_0, input0_stride);
    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        for (uint32_t j = 0; j < groups; j++) {
            noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
            l1_read_addr += input0_stride;
            l1_write_addr += input0_stride;
            l1_write_addr += input1_stride;  // advance past the other input's tiles
        }
    }

    l1_write_addr = base_l1_write_addr + input0_stride;  // move pointer to end of first input
    l1_read_addr = base_l1_read_addr_1;
    noc_async_read_one_packet_set_state(noc_addr_1, input1_stride);
    for (uint32_t i = 0; i < input1_num_tiles_height; i++) {
        for (uint32_t j = 0; j < groups; j++) {
            DPRINT << "reading from " << l1_read_addr << " and writing to " << l1_write_addr << ENDL();
            noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
            l1_read_addr += input1_stride;
            l1_write_addr += input1_stride;
            l1_write_addr += input0_stride;  // advance past the other input's tiles
        }
    }

    noc_async_read_barrier();
}
