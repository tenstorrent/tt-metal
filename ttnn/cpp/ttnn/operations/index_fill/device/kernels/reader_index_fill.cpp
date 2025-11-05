// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <algorithm>

bool contains_element(uint32_t* arr, uint32_t size, uint32_t val) {
    return std::find(arr, arr + size, val) != arr + size;
}

void kernel_main() {
    // Compile-time args
    constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t index_total_size = get_compile_time_arg_val(1);
    constexpr uint32_t index_size = get_compile_time_arg_val(2);
    constexpr bool is_last_dim = get_compile_time_arg_val(3) == 1;
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr auto index_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Run-time args
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t index_buffer_address = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);
    uint32_t end_row = get_arg_val<uint32_t>(3);
    uint32_t num_rows_in_dim = get_arg_val<uint32_t>(4);
    uint32_t dim_size = get_arg_val<uint32_t>(5);

    // Derived
    constexpr uint32_t src_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t index_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t fill_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t onepage = 1;

    const auto s0 = TensorAccessor(input_args, input_buffer_address, input_page_size);
    const auto s1 = TensorAccessor(index_args, index_buffer_address, index_total_size);

    // Read the entire index tensor into L1
    cb_reserve_back(index_cb_id, onepage);
    uint32_t index_addr = get_write_ptr(index_cb_id);
    uint64_t index_noc_addr = s1.get_noc_addr(0);
    noc_async_read(index_noc_addr, index_addr, index_total_size);
    noc_async_read_barrier();
    cb_push_back(index_cb_id, onepage);
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(index_addr);

    // Read input tensor pages
    for (uint32_t row_id = start_row; row_id < end_row; ++row_id) {
        // Performance optimization: use pre-filled page instead of input page?
        bool use_filled_page = false;
        if constexpr (!is_last_dim) {
            uint32_t dim_index = (row_id / num_rows_in_dim) % dim_size;
            use_filled_page = contains_element(index_ptr, index_size, dim_index);
        }

        if (!use_filled_page) {
            // Read input page
            cb_reserve_back(src_cb_id, onepage);
            uint32_t input_addr = get_write_ptr(src_cb_id);
            uint64_t input_noc_addr = s0.get_noc_addr(row_id);
            noc_async_read(input_noc_addr, input_addr, input_page_size);
            noc_async_read_barrier();
            cb_push_back(src_cb_id, onepage);
        }
    }
}
