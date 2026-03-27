// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

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
                min(output_column + elements_per_output_subblock - 1, elements_per_tensor_row - 1);
            output_end_column =
                min(next_output_page_end_column,
                    output_end_column);  // output end column should be the minimum of the next output page end column,
                                         // the end column of the next output subblock and the end of the tensor row

            uint32_t output_page_id = row * num_output_pages_in_row + (output_column / elements_per_output_page);
            uint64_t output_page_noc_addr = accessor_dst.get_noc_addr(output_page_id);
            uint64_t output_addr_subblock_offset =
                ((output_column % elements_per_output_page) / elements_per_output_subblock) * bytes_per_output_subblock;
            uint32_t num_bytes_to_write = (output_end_column - output_column + 1) * bytes_per_element;
            cb_wait_front(cb_id_in1, 1);
            uint32_t output_page_read_addr = get_read_ptr(cb_id_in1);
            uint64_t output_noc_addr = accessor_dst.get_noc_addr(output_page_id) + output_addr_subblock_offset;
            noc_async_write(output_page_read_addr, output_page_noc_addr, num_bytes_to_write);
            noc_async_write_barrier();
            cb_pop_front(cb_id_in1, 1);
            output_column = output_end_column + 1;
        }

        // uint32_t input_start_column = 0;
        // uint32_t input_end_column = input_start_column + elements_per_input_subblock -1;
        // uint32_t output_start_column = 0;
        // uint32_t output_end_column = output_start_column + elements_per_output_subblock -1;

        // while (input_start_column < elements_per_tensor_row) {
        //     if (input_end_column >= output_end_column) { // Case where we are finishing writing an output subblock
        //         // uint32_t bytes_to_write_to_output_subblock;
        //         // uint32_t l1_output_subblock_write_addr_offset;
        //         // uint32_t l1_input_subblock_read_addr_offset;
        //         // if (output_start_column >= input_start_column) {
        //         //     cb_reserve_back(cb_id_in1, 1);
        //         //     bytes_to_write_to_output_subblock = (output_end_column - output_start_column + 1) *
        //         bytes_per_element;
        //         //     l1_output_subblock_write_addr_offset = 0;
        //         //     l1_input_subblock_read_addr_offset = (output_start_column - input_start_column) *
        //         bytes_per_element;
        //         // } else {
        //         //     bytes_to_write_to_output_subblock = (output_end_column - input_start_column + 1) *
        //         bytes_per_element;
        //         //     l1_output_subblock_write_addr_offset = (input_start_column - output_start_column) *
        //         bytes_per_element;
        //         //     l1_input_subblock_read_addr_offset = 0;
        //         // }
        //         // uint32_t l1_output_subblock_write_addr = get_write_ptr(cb_id_in1); // write the output subblock to
        //         the output cb
        //         // tt::data_movement::common::tt_memmove<false, false, true, 0>(
        //         //     l1_output_subblock_write_addr,
        //         //     input_l1_write_addr + l1_input_subblock_read_addr_offset,
        //         //     bytes_to_write_to_output_subblock);
        //         uint32_t output_page_id = row * num_output_pages_in_row + (output_start_column /
        //         elements_per_output_page); uint32_t output_page_offset = (output_start_column %
        //         elements_per_output_page) * bytes_per_element; uint32_t output_page_bytes = (output_end_column -
        //         output_start_column + 1) * bytes_per_element;

        //         if (input_end_column ==  output_end_column) {
        //             // We have processed the entire input subblock, so we must update the start and end indices of
        //             the input subblock as well input_start_column = input_end_column + 1; uint32_t
        //             next_input_page_end_column = input_end_column + (elements_per_input_page - (input_end_column %
        //             elements_per_input_page) - 1); input_end_column = min(input_start_column +
        //             elements_per_input_subblock - 1, elements_per_tensor_row - 1); input_end_column =
        //             min(next_input_page_end_column, input_end_column); // input end column should be the minimum of
        //             the next input page end column, the end column of the next input subblock and the end of the
        //             tensor row
        //         }
        //         // We have processed the entire output subblock, so we must update the start and end indices of the
        //         output subblock output_start_column = output_end_column + 1; uint32_t next_output_page_end_column =
        //         output_end_column + (elements_per_output_page - (output_end_column % elements_per_output_page) - 1);
        //         output_end_column = min(output_start_column + elements_per_output_subblock - 1,
        //         elements_per_tensor_row - 1); output_end_column = min(next_output_page_end_column,
        //         output_end_column); // output end column should be the minimum of the next output page end column,
        //         the end column of the next output subblock and the end of the tensor row
        //         // We have processed the entire output subblock, so we must commit it to the output cb
        //         cb_push_back(cb_id_in1, 1);
        //     } else {
        //         // uint32_t bytes_to_write_to_output_subblock;
        //         // uint32_t l1_output_subblock_write_addr_offset;
        //         // uint32_t l1_input_subblock_read_addr_offset;
        //         // if (output_start_column >= input_start_column) {
        //         //     cb_reserve_back(cb_id_in1, 1); // We are writing a new output subblock, so we need to reserve
        //         a slot on the output cb
        //         //     bytes_to_write_to_output_subblock = (input_end_column - output_start_column + 1) *
        //         bytes_per_element;
        //         //     l1_output_subblock_write_addr_offset = 0;
        //         //     l1_input_subblock_read_addr_offset = (output_start_column - input_start_column) *
        //         bytes_per_element;
        //         // } else {
        //         //     bytes_to_write_to_output_subblock = (input_end_column - input_start_column + 1) *
        //         bytes_per_element;
        //         //     l1_output_subblock_write_addr_offset = (input_start_column - output_start_column) *
        //         bytes_per_element;
        //         //     l1_input_subblock_read_addr_offset = 0;
        //         // }

        //         // uint32_t l1_output_subblock_write_addr = get_write_ptr(cb_id_in1); // Write the output subblock to
        //         the output cb
        //         // tt::data_movement::common::tt_memmove<false, false, true, 0>(
        //         //     l1_output_subblock_write_addr + l1_output_subblock_write_addr_offset,
        //         //     input_l1_write_addr + l1_input_subblock_read_addr_offset,
        //         //     bytes_to_write_to_output_subblock);

        //         // We have processed the entire input subblock, so we must update the start and end indices of the
        //         input subblock as well input_start_column = input_end_column + 1; uint32_t next_input_page_end_column
        //         = input_end_column + (elements_per_input_page - (input_end_column % elements_per_input_page) - 1);
        //         input_end_column = min(input_start_column + elements_per_input_subblock - 1, elements_per_tensor_row
        //         - 1); input_end_column = min(next_input_page_end_column, input_end_column); // input end column
        //         should be the minimum of the next input page end column, the end column of the next input subblock
        //         and the end of the tensor row
        //     }
    }

    // input start column and end column of
    // output start column and end column

    // num sub blocks per output page
    // for each subblock, get first column and last column.
    // using this info, get which input pages and input subblocks to read.
}
