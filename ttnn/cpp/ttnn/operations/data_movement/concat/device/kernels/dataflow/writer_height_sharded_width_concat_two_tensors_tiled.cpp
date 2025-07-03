// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <debug/dprint.h>

void kernel_main() {
    constexpr uint32_t input0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t input0_transpose_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input1_transpose_cb = get_compile_time_arg_val(3);
    constexpr uint32_t concat_cb = get_compile_time_arg_val(4);
    constexpr uint32_t output_transpose_cb = get_compile_time_arg_val(5);
    constexpr uint32_t output_cb = get_compile_time_arg_val(6);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(7);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(8);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(9);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(10);

    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t groups = get_compile_time_arg_val(12);

    constexpr uint32_t input0_stride = tile_size * input0_num_tiles_width / groups;
    constexpr uint32_t input1_stride = tile_size * input1_num_tiles_width / groups;

    constexpr uint32_t width_len_bytes = tile_size * (input0_num_tiles_width + input1_num_tiles_width);

    const uint32_t base_l1_write_addr = get_write_ptr(output_cb);
    uint32_t l1_write_addr = base_l1_write_addr;
    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        cb_reserve_back(output_cb, input0_num_tiles_width + input1_num_tiles_width);
        cb_wait_front(output_transpose_cb, input0_num_tiles_width + input1_num_tiles_width);

        const uint32_t base_l1_read_addr_0 = get_read_ptr(output_transpose_cb);
        const uint64_t noc_addr_0 = get_noc_addr(base_l1_read_addr_0);
        noc_async_read_one_packet_set_state(noc_addr_0, width_len_bytes);
        noc_async_read_one_packet_with_state<true>(base_l1_read_addr_0, l1_write_addr);
        l1_write_addr += width_len_bytes;

        noc_async_read_barrier();

        cb_pop_front(output_transpose_cb, input0_num_tiles_width + input1_num_tiles_width);
        cb_push_back(output_cb, input0_num_tiles_width + input1_num_tiles_width);
    }
}
