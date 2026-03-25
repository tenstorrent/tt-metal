// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

inline uint32_t get_valid_output_page_bytes_to_write(
    uint32_t output_page_id,
    uint32_t num_output_pages_in_row,
    uint32_t elements_per_output_page,
    uint32_t elements_per_tensor_row,
    uint32_t bytes_per_element) {
    const uint32_t col = (output_page_id % num_output_pages_in_row) * elements_per_output_page;
    const uint32_t output_end_col = (col + elements_per_output_page) < elements_per_tensor_row
                                        ? (col + elements_per_output_page - 1)
                                        : elements_per_tensor_row - 1;
    return (output_end_col - col + 1) * bytes_per_element;
}

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(1);

    // compile-time args
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);
    constexpr uint32_t num_output_pages_in_row = get_compile_time_arg_val(3);
    constexpr uint32_t elements_per_output_page = get_compile_time_arg_val(4);
    constexpr uint32_t bytes_per_element = get_compile_time_arg_val(5);
    constexpr uint32_t elements_per_tensor_row = get_compile_time_arg_val(6);

    constexpr auto dst_args = TensorAccessorArgs<7>();
    const auto accessor_dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_dst.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end(); page_iter++) {
            auto output_page_id = page_iter->page_id();

            const auto valid_output_page_bytes_to_write = get_valid_output_page_bytes_to_write(
                output_page_id,
                num_output_pages_in_row,
                elements_per_output_page,
                elements_per_tensor_row,
                bytes_per_element);

            cb_wait_front(cb_id_in1, 1);
            const uint64_t output_page_noc_addr = page_iter->noc_addr();
            uint32_t output_page_read_addr = get_read_ptr(cb_id_in1);

            noc_async_write(output_page_read_addr, output_page_noc_addr, valid_output_page_bytes_to_write);
            noc_async_write_barrier();
            cb_pop_front(cb_id_in1, 1);
        }
    }
}
