// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t stick_size = get_arg_val<uint32_t>(1);
    const uint32_t block_height = get_arg_val<uint32_t>(2);
    const uint32_t block_width_bytes = get_arg_val<uint32_t>(3);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(4);
    const uint32_t input_width_offset_bytes = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);

    constexpr bool dst0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);

    const auto s0 = get_interleaved_addr_gen<dst0_is_dram, dst_stick_size_is_pow2>(
        dst_addr + input_width_offset_bytes, stick_size, dst_log_base_2_of_page_size);

    uint32_t stick_id = start_id;
    cb_wait_front(cb_id_out0, block_height);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
    for (uint32_t h = 0; h < block_height; ++h) {
        uint64_t dst_noc_addr = get_noc_addr(stick_id, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, block_width_bytes);
        stick_id++;
        l1_read_addr += padded_block_width_bytes;
        noc_async_write_barrier();
    }
    cb_pop_front(cb_id_out0, block_height);
}
