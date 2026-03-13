// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_padded_aligned = get_compile_time_arg_val(2);
    constexpr uint32_t num_output_pages_in_row = get_compile_time_arg_val(3);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t output_aligned_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t size_of_valid_data_in_last_output_page_in_row = get_compile_time_arg_val(6);
    constexpr auto dst_args = TensorAccessorArgs<7>();

    const auto s = TensorAccessor(dst_args, dst_addr);

    uint32_t i_page = start_page_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_wait_front(cb_out0, num_sticks_per_barrier);

        uint32_t l1_read_addr = get_read_ptr(cb_out0);

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            uint32_t tmp_addr =
                l1_read_addr;  // AL: maybe change the inner loop to use and update a tmp_addr instead of l1_read_addr
            for (uint32_t p = 0; p < num_output_pages_in_row - 1; p++) {
                uint64_t write_noc_addr = s.get_noc_addr(i_page + p);
                noc_async_write(tmp_addr, write_noc_addr, output_page_size);
                tmp_addr += output_page_size;
            }
            uint64_t write_noc_addr = s.get_noc_addr(i_page + num_output_pages_in_row - 1);
            noc_async_write(tmp_addr, write_noc_addr, size_of_valid_data_in_last_output_page_in_row);
            l1_read_addr += stick_size_padded_aligned;
            i_page += num_output_pages_in_row;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out0, num_sticks_per_barrier);
    }
}
