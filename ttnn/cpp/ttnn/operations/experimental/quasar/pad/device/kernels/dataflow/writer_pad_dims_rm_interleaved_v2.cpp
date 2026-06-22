// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

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
    constexpr uint32_t size_of_valid_data_in_last_output_page_in_row = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    const auto s = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    CircularBuffer cb_out0_exp(cb_out0);

    uint32_t i_page = start_page_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_out0_exp.wait_front(num_sticks_per_barrier);

        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            uint32_t tmp_offset = l1_read_offset;
            for (uint32_t p = 0; p < num_output_pages_in_row - 1; p++) {
                noc.async_write(
                    cb_out0_exp,
                    s,
                    output_page_size,
                    {.offset_bytes = tmp_offset},
                    {.page_id = i_page + p, .offset_bytes = 0});
                tmp_offset += output_page_size;
            }
            noc.async_write(
                cb_out0_exp,
                s,
                size_of_valid_data_in_last_output_page_in_row,
                {.offset_bytes = tmp_offset},
                {.page_id = i_page + num_output_pages_in_row - 1, .offset_bytes = 0});
            l1_read_offset += stick_size_padded_aligned;
            i_page += num_output_pages_in_row;
        }
        noc.async_write_barrier();
        cb_out0_exp.pop_front(num_sticks_per_barrier);
    }
}
