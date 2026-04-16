// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t stick_size_offset = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(3);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(4);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(5);
    uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages_in_row = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t size_of_valid_data_in_last_page_in_row = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const auto s0 = TensorAccessor(dst_args, dst_addr);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_wait_front(cb_id_out0, num_read_per_barrier);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            if (num_pages_in_row == 1) {
                uint64_t dst_noc_addr = s0.get_noc_addr(i_stick);
                noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
                l1_read_addr += stick_size_offset;
                i_stick += 1;
            } else {
                uint32_t row_l1 = l1_read_addr;
                for (uint32_t p = 0; p < num_pages_in_row - 1; p++) {
                    uint64_t dst_noc_addr = s0.get_noc_addr(i_stick);
                    noc_async_write(row_l1, dst_noc_addr, page_size);
                    noc_async_write_barrier();
                    row_l1 += page_size;
                    i_stick += 1;
                }
                uint64_t dst_noc_addr = s0.get_noc_addr(i_stick);
                noc_async_write(row_l1, dst_noc_addr, size_of_valid_data_in_last_page_in_row);
                noc_async_write_barrier();
                i_stick += 1;
                l1_read_addr += stick_size_offset;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_read_per_barrier);
    }
}
