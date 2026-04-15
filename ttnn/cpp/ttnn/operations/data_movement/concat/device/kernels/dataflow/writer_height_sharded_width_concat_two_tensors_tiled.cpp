// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/debug/dprint.h>
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t input0_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input0_transpose_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input1_transpose_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t concat_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t output_transpose_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(6);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(7);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(8);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(9);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(10);

    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t groups = get_compile_time_arg_val(12);

    constexpr uint32_t width_len_bytes = tile_size * (input0_num_tiles_width + input1_num_tiles_width);

    experimental::CircularBuffer output_cb(output_cb_id);
    experimental::CircularBuffer output_transpose_cb(output_transpose_cb_id);

    const uint32_t base_l1_write_addr = output_cb.get_write_ptr();
    uint32_t l1_write_addr = base_l1_write_addr;
    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        output_cb.reserve_back(input0_num_tiles_width + input1_num_tiles_width);
        output_transpose_cb.wait_front(input0_num_tiles_width + input1_num_tiles_width);

        const uint32_t base_l1_read_addr_0 = output_transpose_cb.get_read_ptr();
        const uint64_t noc_addr_0 = get_noc_addr(base_l1_read_addr_0);
        noc_async_read(noc_addr_0, l1_write_addr, width_len_bytes);
        l1_write_addr += width_len_bytes;

        noc_async_read_barrier();

        output_transpose_cb.pop_front(input0_num_tiles_width + input1_num_tiles_width);
        output_cb.push_back(input0_num_tiles_width + input1_num_tiles_width);
    }
}
