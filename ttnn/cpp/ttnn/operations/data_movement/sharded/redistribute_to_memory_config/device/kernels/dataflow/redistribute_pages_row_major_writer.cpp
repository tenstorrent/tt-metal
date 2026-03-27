// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t u32_min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t num_output_pages_in_row = get_compile_time_arg_val(1);
    constexpr uint32_t elements_per_output_page = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_element = get_compile_time_arg_val(3);
    constexpr uint32_t elements_per_tensor_row = get_compile_time_arg_val(4);
    constexpr uint32_t bytes_per_output_subblock = get_compile_time_arg_val(5);

    constexpr auto dst_args = TensorAccessorArgs<6>();
    const auto accessor_dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t elements_per_output_subblock = bytes_per_output_subblock / bytes_per_element;

    for (uint32_t row = start_row; row < start_row + num_rows_to_process; ++row) {
        uint32_t output_column = 0;
        while (output_column < elements_per_tensor_row) {
            uint32_t next_output_page_end_column =
                output_column + (elements_per_output_page - (output_column % elements_per_output_page) - 1);
            uint32_t output_end_column =
                u32_min(output_column + elements_per_output_subblock - 1, elements_per_tensor_row - 1);
            output_end_column = u32_min(
                next_output_page_end_column,
                output_end_column);  // output end column should be the minimum of the next output page end column,
                                     // the end column of the next output subblock and the end of the tensor row

            uint32_t output_page_id = row * num_output_pages_in_row + (output_column / elements_per_output_page);
            uint64_t output_page_noc_addr = accessor_dst.get_noc_addr(output_page_id);
            uint64_t output_addr_subblock_offset =
                ((output_column % elements_per_output_page) / elements_per_output_subblock) * bytes_per_output_subblock;
            uint32_t num_bytes_to_write = (output_end_column - output_column + 1) * bytes_per_element;
            cb_wait_front(cb_id_in1, 1);
            uint32_t output_page_read_addr = get_read_ptr(cb_id_in1);
            uint64_t output_subblock_noc_addr = accessor_dst.get_noc_addr(output_page_id) + output_addr_subblock_offset;
            noc_async_write(output_page_read_addr, output_subblock_noc_addr, num_bytes_to_write);
            noc_async_write_barrier();
            cb_pop_front(cb_id_in1, 1);
            output_column = output_end_column + 1;
        }
    }
}
