// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

FORCE_INLINE uint32_t u32_min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

void kernel_main() {
    // run-time args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t num_output_pages_in_row = get_compile_time_arg_val(2);
    constexpr uint32_t num_input_pages_in_row = get_compile_time_arg_val(3);
    constexpr uint32_t elements_per_output_page = get_compile_time_arg_val(4);
    constexpr uint32_t bytes_per_element = get_compile_time_arg_val(5);
    constexpr uint32_t elements_per_input_page = get_compile_time_arg_val(6);
    constexpr uint32_t elements_per_tensor_row = get_compile_time_arg_val(7);
    constexpr uint32_t bytes_per_input_subblock = get_compile_time_arg_val(8);
    constexpr uint32_t bytes_per_output_subblock = get_compile_time_arg_val(9);

    constexpr auto src_args = TensorAccessorArgs<10>();
    const auto accessor_src = TensorAccessor(src_args, src_addr);

    const uint32_t elements_per_output_subblock = bytes_per_output_subblock / bytes_per_element;
    const uint32_t elements_per_input_subblock = bytes_per_input_subblock / bytes_per_element;
    cb_reserve_back(cb_id_in0, 1);
    const uint32_t input_l1_write_addr = get_write_ptr(cb_id_in0);

    for (uint32_t row = start_row; row < start_row + num_rows_to_process; ++row) {
        uint32_t input_start_column = 0;
        uint32_t input_end_column = input_start_column + elements_per_input_subblock - 1;
        uint32_t output_start_column = 0;
        uint32_t output_end_column = output_start_column + elements_per_output_subblock - 1;
        while (input_start_column < elements_per_tensor_row) {
            if (input_start_column >= output_start_column) {  // We need to read in a new input subblock
                uint32_t input_page_id = row * num_input_pages_in_row + (input_start_column / elements_per_input_page);
                uint64_t input_page_noc_addr = accessor_src.get_noc_addr(input_page_id);
                uint64_t input_subblock_offset =
                    ((input_start_column % elements_per_input_page) / elements_per_input_subblock) *
                    bytes_per_input_subblock;
                uint32_t num_bytes_to_read = (input_end_column - input_start_column + 1) * bytes_per_element;
                noc_async_read(input_page_noc_addr + input_subblock_offset, input_l1_write_addr, num_bytes_to_read);
                noc_async_read_barrier();
            }
            if (input_end_column >= output_end_column) {  // Case where we are finishing writing an output subblock
                uint32_t bytes_to_write_to_output_subblock;
                uint32_t l1_output_subblock_write_addr_offset;
                uint32_t l1_input_subblock_read_addr_offset;
                if (output_start_column >= input_start_column) {
                    cb_reserve_back(
                        cb_id_in1,
                        1);  // We are writing a new output subblock, so we need to reserve a slot on the output cb
                    bytes_to_write_to_output_subblock =
                        (output_end_column - output_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset = 0;
                    l1_input_subblock_read_addr_offset =
                        (output_start_column - input_start_column) *
                        bytes_per_element;  // part of the input subblock was already read in previous iterations
                } else {
                    bytes_to_write_to_output_subblock =
                        (output_end_column - input_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset =
                        (input_start_column - output_start_column) *
                        bytes_per_element;  // part of the output subblock was already written in previous iterations
                    l1_input_subblock_read_addr_offset = 0;
                }
                // cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_output_subblock_write_addr =
                    get_write_ptr(cb_id_in1);  // write the output subblock to the output cb
                tt::data_movement::common::tt_memmove<false, false, true, 0>(
                    l1_output_subblock_write_addr + l1_output_subblock_write_addr_offset,
                    input_l1_write_addr + l1_input_subblock_read_addr_offset,
                    bytes_to_write_to_output_subblock);

                if (input_end_column == output_end_column) {
                    // We have processed the entire input subblock, so we must update the start and end indices of the
                    // input subblock as well
                    input_start_column = input_end_column + 1;
                    uint32_t next_input_page_end_column =
                        input_start_column +
                        (elements_per_input_page - (input_start_column % elements_per_input_page) - 1);
                    input_end_column =
                        u32_min(input_start_column + elements_per_input_subblock - 1, elements_per_tensor_row - 1);
                    input_end_column = u32_min(
                        next_input_page_end_column,
                        input_end_column);  // input end column should be the minimum of the next input page end column,
                                            // the end column of the next input subblock and the end of the tensor row
                }
                // We have processed the entire output subblock, so we must update the start and end indices of the
                // output subblock
                output_start_column = output_end_column + 1;
                uint32_t next_output_page_end_column =
                    output_start_column +
                    (elements_per_output_page - (output_start_column % elements_per_output_page) - 1);
                output_end_column =
                    u32_min(output_start_column + elements_per_output_subblock - 1, elements_per_tensor_row - 1);
                output_end_column = u32_min(
                    next_output_page_end_column,
                    output_end_column);  // output end column should be the minimum of the next output page end column,
                                         // the end column of the next output subblock and the end of the tensor row
                // We have processed the entire output subblock, so we must commit it to the output cb
                cb_push_back(cb_id_in1, 1);
            } else {  // Case where we are finishing reading in an input subblock
                uint32_t bytes_to_write_to_output_subblock;
                uint32_t l1_output_subblock_write_addr_offset;
                uint32_t l1_input_subblock_read_addr_offset;
                if (output_start_column >= input_start_column) {
                    cb_reserve_back(
                        cb_id_in1,
                        1);  // We are writing a new output subblock, so we need to reserve a slot on the output cb
                    bytes_to_write_to_output_subblock =
                        (input_end_column - output_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset = 0;
                    l1_input_subblock_read_addr_offset = (output_start_column - input_start_column) * bytes_per_element;
                } else {
                    bytes_to_write_to_output_subblock = (input_end_column - input_start_column + 1) * bytes_per_element;
                    l1_output_subblock_write_addr_offset =
                        (input_start_column - output_start_column) * bytes_per_element;
                    l1_input_subblock_read_addr_offset = 0;
                }

                uint32_t l1_output_subblock_write_addr =
                    get_write_ptr(cb_id_in1);  // Write the output subblock to the output cb
                tt::data_movement::common::tt_memmove<false, false, true, 0>(
                    l1_output_subblock_write_addr + l1_output_subblock_write_addr_offset,
                    input_l1_write_addr + l1_input_subblock_read_addr_offset,
                    bytes_to_write_to_output_subblock);

                // We have processed the entire input subblock, so we must update the start and end indices of the input
                // subblock as well
                input_start_column = input_end_column + 1;
                uint32_t next_input_page_end_column =
                    input_start_column + (elements_per_input_page - (input_start_column % elements_per_input_page) - 1);
                input_end_column =
                    u32_min(input_start_column + elements_per_input_subblock - 1, elements_per_tensor_row - 1);
                input_end_column = u32_min(
                    next_input_page_end_column,
                    input_end_column);  // input end column should be the minimum of the next input page end column,
                                        // the end column of the next input subblock and the end of the tensor row
            }
        }
    }

    cb_push_back(cb_id_in0, 1);
    cb_wait_front(cb_id_in0, 1);
    cb_pop_front(cb_id_in0, 1);
}
