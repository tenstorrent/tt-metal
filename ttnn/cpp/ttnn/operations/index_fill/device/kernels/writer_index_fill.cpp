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
    constexpr uint32_t output_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t index_size = get_compile_time_arg_val(1);
    constexpr uint32_t elem_size = get_compile_time_arg_val(2);
    constexpr bool is_last_dim = get_compile_time_arg_val(3) == 1;
    constexpr auto dst_args = TensorAccessorArgs<4>();

    // Run-time args
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t start_row = get_arg_val<uint32_t>(1);
    uint32_t end_row = get_arg_val<uint32_t>(2);
    uint32_t num_rows_in_dim = get_arg_val<uint32_t>(3);
    uint32_t dim_size = get_arg_val<uint32_t>(4);
    uint32_t fill_value = get_arg_val<uint32_t>(5);

    // Derived
    static_assert(elem_size == 2 || elem_size == 4, "Unsupported elem_size");
    using IntType = std::conditional_t<(elem_size == 2), uint16_t, uint32_t>;

    constexpr uint32_t src_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t index_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t fill_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t onepage = 1;
    constexpr uint32_t row_size = output_page_size / elem_size;

    const auto s = TensorAccessor(dst_args, output_buffer_address, output_page_size);

    // Performance optimization:
    // Prefill an input page (in L1) with fill_value
    cb_reserve_back(fill_cb_id, onepage);
    uint32_t fill_addr = get_write_ptr(fill_cb_id);
    auto* fill_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(fill_addr);
    if constexpr (!is_last_dim) {
        for (uint32_t i = 0; i < row_size; ++i) {
            fill_ptr[i] = fill_value;
        }
    }
    cb_push_back(fill_cb_id, onepage);

    // Wait for index tensor to be available
    cb_wait_front(index_cb_id, onepage);
    uint32_t index_addr = get_read_ptr(index_cb_id);
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(index_addr);

    // Write input pages
    for (uint32_t row_id = start_row; row_id < end_row; ++row_id) {
        // Performance optimization: use pre-filled page instead of input page?
        bool use_filled_page = false;
        if constexpr (!is_last_dim) {
            uint32_t dim_index = (row_id / num_rows_in_dim) % dim_size;
            use_filled_page = contains_element(index_ptr, index_size, dim_index);
        }

        if (use_filled_page) {
            // Write filled page to output tensor
            uint64_t output_noc_addr = s.get_noc_addr(row_id);
            noc_async_write(fill_addr, output_noc_addr, output_page_size);
            noc_async_write_barrier();
        } else {
            // Wait for input page from reader kernel
            cb_wait_front(src_cb_id, onepage);
            uint32_t input_addr = get_read_ptr(src_cb_id);

            // If dim is the last dim, we need to fill in certain spots in the page
            if constexpr (is_last_dim) {
                auto* input_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(input_addr);
                for (uint32_t i = 0; i < index_size; ++i) {
                    input_ptr[index_ptr[i]] = fill_value;
                }
            }

            // Write page to output tensor
            uint64_t output_noc_addr = s.get_noc_addr(row_id);
            noc_async_write(input_addr, output_noc_addr, output_page_size);
            noc_async_write_barrier();

            cb_pop_front(src_cb_id, onepage);
        }
    }
}
