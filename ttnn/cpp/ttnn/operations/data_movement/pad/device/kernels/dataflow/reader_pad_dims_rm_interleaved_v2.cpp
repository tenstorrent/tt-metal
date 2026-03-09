// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

inline __attribute__((always_inline)) void fill_pad_cb_with_val(
    const uint32_t cb_id, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_page_id = get_arg_val<uint32_t>(3);
    uint32_t front_pad_n = get_arg_val<uint32_t>(4);
    uint32_t front_pad_c = get_arg_val<uint32_t>(5);
    uint32_t front_pad_h = get_arg_val<uint32_t>(6);
    tt_l1_ptr uint32_t* start_dim_offset = (tt_l1_ptr uint32_t*)(get_arg_addr(7));

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t N_padded = get_compile_time_arg_val(4);
    constexpr uint32_t H_padded = get_compile_time_arg_val(5);
    constexpr uint32_t C_padded = get_compile_time_arg_val(6);
    constexpr uint32_t stick_size_padded = get_compile_time_arg_val(7);
    constexpr uint32_t stick_size_padded_front = get_compile_time_arg_val(8);
    constexpr uint32_t stick_size_padded_end = get_compile_time_arg_val(9);
    constexpr uint32_t num_zero_pad_sticks_read = get_compile_time_arg_val(10);
    constexpr uint32_t last_zero_stick_size = get_compile_time_arg_val(11);
    constexpr uint32_t stick_size_padded_aligned = get_compile_time_arg_val(18);

    constexpr bool not_pad_by_zero = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t front_padding = get_compile_time_arg_val(8);
    constexpr bool unaligned = get_compile_time_arg_val(19) == 1;

    constexpr uint32_t num_input_pages_in_row = get_compile_time_arg_val(20);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t input_aligned_page_size = get_compile_time_arg_val(22);
    constexpr uint32_t size_of_valid_data_in_last_input_page_in_row = get_compile_time_arg_val(23);
    constexpr auto src_args = TensorAccessorArgs<24>();

    uint32_t packed_pad_value = 0;
    uint32_t row_major_min_bytes = 0;
    uint32_t num_front_pad_sticks_read = 0;
    uint32_t num_end_pad_sticks_read = 0;
    uint32_t num_sticks_padded_read = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = kernel_compile_time_args[13];
        row_major_min_bytes = kernel_compile_time_args[14];
        num_front_pad_sticks_read = kernel_compile_time_args[15];
        num_end_pad_sticks_read = kernel_compile_time_args[16];
        num_sticks_padded_read = kernel_compile_time_args[17];
    }

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_pad = tt::CBIndex::c_1;
    constexpr uint32_t cb_pad_align = tt::CBIndex::c_2;

    const auto s = TensorAccessor(src_args, src_addr);

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint64_t pad_val_addr = get_read_ptr(cb_pad);
    uint64_t pad_val_noc_addr = get_noc_addr(pad_val_addr);

    uint64_t pad_align_addr = get_read_ptr(cb_pad_align);
    uint64_t pad_align_write_addr = get_write_ptr(cb_pad_align);
    uint64_t pad_align_noc_addr = get_noc_addr(pad_align_addr);

    fill_pad_cb_with_val(cb_pad, stick_size_padded, packed_pad_value);

    uint32_t i_page = start_page_id;
    uint32_t curr_c = start_dim_offset[2], curr_h = start_dim_offset[1], curr_n = start_dim_offset[3];
    DPRINT << "reader_pad_dims_rm_interleaved_v2: curr_c: " << curr_c << ", curr_h: " << curr_h
           << ", curr_n: " << curr_n << ENDL();
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_reserve_back(cb_in0, num_sticks_per_barrier);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                              (curr_n >= front_pad_n and curr_n < N);
            noc_async_read(pad_val_noc_addr, l1_write_addr, stick_size_padded);
            noc_async_read_barrier();
            if (read_stick) {
                if constexpr (front_padding) {  // Read noc into cb_pad_align l1
                    uint32_t temp_addr = get_write_ptr(cb_pad_align);
                    for (uint32_t p = 0; p < num_input_pages_in_row - 1; p++) {
                        uint64_t page_noc_addr = s.get_noc_addr(i_page + p);
                        noc_async_read(page_noc_addr, temp_addr, input_page_size);
                        temp_addr += input_page_size;
                    }
                    uint64_t last_page_noc_addr = s.get_noc_addr(i_page + num_input_pages_in_row - 1);
                    noc_async_read(last_page_noc_addr, temp_addr, size_of_valid_data_in_last_input_page_in_row);
                    noc_async_read_barrier();
                    memmove(
                        (void*)(l1_write_addr + stick_size_padded_front),
                        (void*)(get_read_ptr(cb_pad_align)),
                        (size_t)(stick_size_bytes));
                } else if constexpr (unaligned) {
                    uint32_t temp_addr = get_write_ptr(cb_pad_align);
                    for (uint32_t p = 0; p < num_input_pages_in_row - 1; p++) {
                        uint64_t page_noc_addr = s.get_noc_addr(i_page + p);
                        noc_async_read(page_noc_addr, temp_addr, input_page_size);
                        temp_addr += input_page_size;
                    }
                    uint64_t last_page_noc_addr = s.get_noc_addr(i_page + num_input_pages_in_row - 1);
                    noc_async_read(last_page_noc_addr, temp_addr, size_of_valid_data_in_last_input_page_in_row);
                    noc_async_read_barrier();
                    noc_async_read(pad_align_noc_addr, l1_write_addr, stick_size_bytes);
                } else {
                    uint32_t write_addr = l1_write_addr;
                    for (uint32_t p = 0; p < num_input_pages_in_row - 1; p++) {
                        uint64_t page_noc_addr = s.get_noc_addr(i_page + p);
                        noc_async_read(page_noc_addr, write_addr, input_page_size);
                        write_addr += input_page_size;
                    }
                    uint64_t last_page_noc_addr = s.get_noc_addr(i_page + num_input_pages_in_row - 1);
                    noc_async_read(last_page_noc_addr, write_addr, size_of_valid_data_in_last_input_page_in_row);
                }
                i_page += num_input_pages_in_row;
            }
            l1_write_addr += stick_size_padded_aligned;
            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, num_sticks_per_barrier);
    }
    DPRINT << "reader_pad_dims_rm_interleaved_v2: end" << ENDL();
}
