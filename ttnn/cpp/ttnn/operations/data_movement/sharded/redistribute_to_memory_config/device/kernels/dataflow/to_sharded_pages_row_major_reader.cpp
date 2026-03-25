// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

struct InputPageReadInfo {
    uint32_t input_first_page_id;
    uint32_t input_first_page_offset;
    uint32_t input_first_page_bytes;
    uint32_t input_last_page_id;
    uint32_t input_last_page_bytes;
    uint32_t output_page_valid_data_bytes;
};

inline uint32_t div_up_u32(uint32_t value, uint32_t divisor) { return (value + divisor - 1) / divisor; }

inline InputPageReadInfo get_input_page_read_info(
    uint32_t output_page_id,
    uint32_t num_output_pages_in_row,
    uint32_t num_input_pages_in_row,
    uint32_t elements_per_output_page,
    uint32_t elements_per_tensor_row,
    uint32_t bytes_per_element,
    uint32_t elements_per_input_page,
    uint32_t bytes_per_input_page) {
    const uint32_t row = output_page_id / num_output_pages_in_row;
    const uint32_t col = (output_page_id % num_output_pages_in_row) * elements_per_output_page;
    const uint32_t input_first_page_offset = (col % elements_per_input_page) * bytes_per_element;
    const uint32_t output_end_col = (col + elements_per_output_page) < elements_per_tensor_row
                                        ? (col + elements_per_output_page - 1)
                                        : elements_per_tensor_row - 1;

    const uint32_t input_first_page_id = row * num_input_pages_in_row + (col / elements_per_input_page);
    const uint32_t input_first_page_bytes = bytes_per_input_page - input_first_page_offset;
    const uint32_t input_last_page_id = row * num_input_pages_in_row + (output_end_col / elements_per_input_page);
    const uint32_t input_last_page_bytes = (input_last_page_id == input_first_page_id)
                                               ? input_first_page_bytes
                                               : ((output_end_col % elements_per_input_page) + 1) * bytes_per_element;

    return {
        .input_first_page_id = input_first_page_id,
        .input_first_page_offset = input_first_page_offset,
        .input_first_page_bytes = input_first_page_bytes,
        .input_last_page_id = input_last_page_id,
        .input_last_page_bytes = input_last_page_bytes,
        .output_page_valid_data_bytes = (output_end_col - col + 1) * bytes_per_element};
}

void kernel_main() {
    // run-time args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t num_shards = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores = get_compile_time_arg_val(3);
    constexpr uint32_t num_output_pages_in_row = get_compile_time_arg_val(4);
    constexpr uint32_t num_input_pages_in_row = get_compile_time_arg_val(5);
    constexpr uint32_t elements_per_output_page = get_compile_time_arg_val(6);
    constexpr uint32_t bytes_per_element = get_compile_time_arg_val(7);
    constexpr uint32_t elements_per_input_page = get_compile_time_arg_val(8);
    constexpr uint32_t elements_per_tensor_row = get_compile_time_arg_val(9);

    constexpr auto src_args = TensorAccessorArgs<10>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto accessor_src = TensorAccessor(src_args, src_addr);
    const auto accessor_dst = TensorAccessor(dst_args, dst_addr);

    constexpr uint32_t input_page_size_bytes = elements_per_input_page * bytes_per_element;
    const uint32_t input_pages_l1_write_addr = get_write_ptr(cb_id_in0);
    cb_reserve_back(cb_id_in0, 1);

    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_dst.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end(); page_iter++) {
            auto output_page_id = page_iter->page_id();

            const auto input_page_read_info = get_input_page_read_info(
                output_page_id,
                num_output_pages_in_row,
                num_input_pages_in_row,
                elements_per_output_page,
                elements_per_tensor_row,
                bytes_per_element,
                elements_per_input_page,
                input_page_size_bytes);
            const auto input_first_page_id = input_page_read_info.input_first_page_id;
            const auto input_first_page_offset = input_page_read_info.input_first_page_offset;
            const auto input_first_page_bytes = input_page_read_info.input_first_page_bytes;
            const auto input_last_page_id = input_page_read_info.input_last_page_id;
            const auto input_last_page_bytes = input_page_read_info.input_last_page_bytes;
            cb_reserve_back(cb_id_in1, 1);
            uint32_t l1_output_page_write_addr = get_write_ptr(cb_id_in1);

            // Read the input pages into the CB
            const auto input_pages_to_read = input_last_page_id - input_first_page_id + 1;

            for (uint32_t input_page_id_offset = 0; input_page_id_offset < input_pages_to_read;
                 ++input_page_id_offset) {
                uint32_t input_page_overlapping_bytes_with_output_page = input_page_size_bytes;
                uint32_t input_page_offset_bytes = 0;
                if (input_page_id_offset == 0) {
                    input_page_overlapping_bytes_with_output_page =
                        input_first_page_bytes;  // We may only be overlapping with the first part of the last input
                                                 // page
                    input_page_offset_bytes = input_first_page_offset;
                } else if (input_page_id_offset == input_pages_to_read - 1) {  // In cases where first_page_id ==
                                                                               // last_page_id, this check is never
                                                                               // reached.
                    input_page_overlapping_bytes_with_output_page =
                        input_last_page_bytes;  // We may only be overlapping with the last part of the first input page
                }
                const uint64_t input_page_noc_addr =
                    accessor_src.get_noc_addr(input_first_page_id + input_page_id_offset);
                noc_async_read(input_page_noc_addr, input_pages_l1_write_addr, input_page_size_bytes);
                noc_async_read_barrier();

                tt::data_movement::common::tt_memmove<false, false, true, 0>(
                    l1_output_page_write_addr,
                    input_pages_l1_write_addr + input_page_offset_bytes,
                    input_page_overlapping_bytes_with_output_page);
                l1_output_page_write_addr += input_page_overlapping_bytes_with_output_page;
            }
            cb_push_back(cb_id_in1, 1);
        }
    }
    cb_push_back(cb_id_in0, 1);
    cb_wait_front(cb_id_in0, 1);
    cb_pop_front(cb_id_in0, 1);
}
