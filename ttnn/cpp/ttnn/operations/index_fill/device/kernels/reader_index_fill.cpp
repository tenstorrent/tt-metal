// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

bool is_in_indices(uint32_t* index_ptr, uint32_t size, uint32_t row_id) {
    for (uint32_t i = 0; i < size; i++) {
        if (row_id == index_ptr[i]) {
            return true;
        }
    }
    return false;
}

void kernel_main() {
    // Compile-time args
    constexpr bool is_last_dim = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t index_size = get_compile_time_arg_val(1);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t index_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t elem_size = get_compile_time_arg_val(4);
    constexpr uint32_t row_size = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto index_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Run-time args
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t index_addr = get_arg_val<uint32_t>(1);
    uint32_t fill_value = get_arg_val<uint32_t>(2);
    uint32_t start_row_id = get_arg_val<uint32_t>(3);
    uint32_t num_rows_per_core = get_arg_val<uint32_t>(4);
    uint32_t num_rows_to_fill_per_index = get_arg_val<uint32_t>(5);
    uint32_t dim = get_arg_val<uint32_t>(6);

    // Derived
    constexpr uint32_t src_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t index_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(input_args, input_addr, input_page_size);
    const auto s1 = TensorAccessor(index_args, index_addr, index_page_size);

    // Read the index tensor into L1
    cb_reserve_back(index_cb_id, onetile);
    uint32_t index_cb_reader = get_write_ptr(index_cb_id);
    uint64_t index_noc_addr = get_noc_addr(0, s1);
    noc_async_read(index_noc_addr, index_cb_reader, index_page_size);
    noc_async_read_barrier();
    uint32_t* index_ptr = reinterpret_cast<uint32_t*>(index_cb_reader);

    // Read the input tensor and fill in values
    uint32_t end_row_id = start_row_id + num_rows_per_core;
    if (is_last_dim) {
        for (uint32_t row_id = start_row_id; row_id < end_row_id; ++row_id) {
            cb_reserve_back(src_cb_id, onetile);
            uint32_t src_cb_reader = get_write_ptr(src_cb_id);
            uint64_t input_noc_addr = get_noc_addr(row_id, s0);
            noc_async_read(input_noc_addr, src_cb_reader, input_page_size);
            noc_async_read_barrier();

            if constexpr (elem_size == 2) {
                uint16_t* input_ptr = reinterpret_cast<uint16_t*>(src_cb_reader);
                for (uint32_t i = 0; i < index_size; i++) {
                    uint32_t current_index = index_ptr[i];
                    input_ptr[current_index] = fill_value;
                }
            } else {
                uint32_t* input_ptr = reinterpret_cast<uint32_t*>(src_cb_reader);
                for (uint32_t i = 0; i < index_size; i++) {
                    uint32_t current_index = index_ptr[i];
                    input_ptr[current_index] = fill_value;
                }
            }

            cb_push_back(src_cb_id, onetile);
        }
    } else {
        for (uint32_t row_id = start_row_id; row_id < end_row_id; ++row_id) {
            cb_reserve_back(src_cb_id, onetile);
            uint32_t src_cb_reader = get_write_ptr(src_cb_id);
            uint64_t input_noc_addr = get_noc_addr(row_id, s0);
            noc_async_read(input_noc_addr, src_cb_reader, input_page_size);
            noc_async_read_barrier();

            if (is_in_indices(index_ptr, index_size, row_id / num_rows_to_fill_per_index % dim)) {
                if constexpr (elem_size == 2) {
                    auto ptr = reinterpret_cast<uint16_t*>(src_cb_reader);
                    for (uint32_t i = 0; i < row_size; ++i) {
                        ptr[i] = fill_value;
                    }
                } else {
                    auto ptr = reinterpret_cast<uint32_t*>(src_cb_reader);
                    for (uint32_t i = 0; i < row_size; ++i) {
                        ptr[i] = fill_value;
                    }
                }
            }
            cb_push_back(src_cb_id, onetile);
        }
    }
    cb_push_back(index_cb_id, onetile);
}
