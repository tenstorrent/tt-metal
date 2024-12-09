// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

#define ENABLE_DEBUG 1

#if ENABLE_DEBUG
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

inline __attribute__((always_inline)) void fill_pad_cb_with_val(
    const uint32_t cb_id, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t front_pad_n = get_arg_val<uint32_t>(4);
    uint32_t front_pad_c = get_arg_val<uint32_t>(5);
    uint32_t front_pad_h = get_arg_val<uint32_t>(6);
    tt_l1_ptr uint32_t* start_dim_offset = (tt_l1_ptr uint32_t*)(get_arg_addr(7));

    DPRINT << "run time args: " << ENDL();
    DPRINT << "num_sticks_per_core_read= " << num_sticks_per_core_read << ENDL();
    DPRINT << "num_read_per_barrier= " << num_read_per_barrier << ENDL();
    DPRINT << "start_id= " << start_id << ENDL();
    DPRINT << "front_pad_n= " << front_pad_n << ENDL();
    DPRINT << "front_pad_c= " << front_pad_c << ENDL();
    DPRINT << "front_pad_h= " << front_pad_h << ENDL();
    DPRINT << "start_dim_offset:dim0= " << start_dim_offset[0] << ", dim1= " << start_dim_offset[1]
           << ", dim2= " << start_dim_offset[2] << ", dim3= " << start_dim_offset[3] << ENDL();

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t C = get_compile_time_arg_val(3);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t N_padded = get_compile_time_arg_val(5);
    constexpr uint32_t H_padded = get_compile_time_arg_val(6);
    constexpr uint32_t C_padded = get_compile_time_arg_val(7);
    constexpr uint32_t stick_size_padded = get_compile_time_arg_val(8);
    constexpr uint32_t stick_size_padded_front = get_compile_time_arg_val(9);
    constexpr uint32_t stick_size_padded_end = get_compile_time_arg_val(10);
    constexpr uint32_t num_zero_pad_sticks_read = get_compile_time_arg_val(11);
    constexpr uint32_t last_zero_stick_size = get_compile_time_arg_val(12);
    DPRINT << "compile time args: " << ENDL();
    if (src_is_dram) {
        DPRINT << "src_is_dram" << ENDL();
    } else {
        DPRINT << "src_is_L1" << ENDL();
    }
    DPRINT << "N= " << N << ENDL();
    DPRINT << "H= " << H << ENDL();
    DPRINT << "C= " << C << ENDL();
    DPRINT << "stick_size_bytes= " << stick_size_bytes << ENDL();
    DPRINT << "N_padded= " << N_padded << ENDL();
    DPRINT << "H_padded= " << H_padded << ENDL();
    DPRINT << "C_padded= " << C_padded << ENDL();
    DPRINT << "stick_size_padded= " << stick_size_padded << ENDL();
    DPRINT << "stick_size_padded_front= " << stick_size_padded_front << ENDL();
    DPRINT << "stick_size_padded_end= " << stick_size_padded_end << ENDL();
    DPRINT << "num_zero_pad_sticks_read= " << num_zero_pad_sticks_read << ENDL();
    DPRINT << "last_zero_stick_size= " << last_zero_stick_size << ENDL();

#define not_pad_by_zero get_compile_time_arg_val(13) == 1
#define front_padding get_compile_time_arg_val(9)
#if (not_pad_by_zero)
    constexpr uint32_t packed_pad_value = get_compile_time_arg_val(14);
    constexpr uint32_t row_major_min_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t num_front_pad_sticks_read = get_compile_time_arg_val(16);
    constexpr uint32_t num_end_pad_sticks_read = get_compile_time_arg_val(17);
    constexpr uint32_t num_sticks_padded_read = get_compile_time_arg_val(18);
    DPRINT << "not pad by zero = " << get_compile_time_arg_val(13) << ENDL();
    DPRINT << "packed_pad_value = " << packed_pad_value << ENDL();
    DPRINT << "row_major_min_bytes= " << row_major_min_bytes << ENDL();
    DPRINT << "num_front_pad_sticks_read= " << num_front_pad_sticks_read << ENDL();
    DPRINT << "num_end_pad_sticks_read= " << num_end_pad_sticks_read << ENDL();
    DPRINT << "num_sticks_padded_read= " << num_sticks_padded_read << ENDL();
#endif

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_pad = tt::CBIndex::c_1;
    constexpr uint32_t cb_pad_align = tt::CBIndex::c_2;

#define stick_size_is_pow2 get_compile_time_arg_val(19) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(20);
#else
    constexpr uint32_t page_size = get_compile_time_arg_val(20);
#endif
#if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src_is_dram> s = {.bank_base_address = src_addr, .page_size = page_size};
#endif

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint64_t pad_val_addr = get_read_ptr(cb_pad);
    uint64_t pad_val_noc_addr = get_noc_addr(pad_val_addr);

    uint64_t pad_align_addr = get_read_ptr(cb_pad_align);
    uint64_t pad_align_write_addr = get_write_ptr(cb_pad_align);
    uint64_t pad_align_noc_addr = get_noc_addr(pad_align_addr);

#if (not_pad_by_zero)
    fill_pad_cb_with_val(cb_pad, stick_size_padded, packed_pad_value);
#endif

    uint32_t i_stick = start_id;
    uint32_t curr_c = start_dim_offset[2], curr_h = start_dim_offset[1], curr_n = start_dim_offset[3];
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_reserve_back(cb_in0, num_read_per_barrier);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                              (curr_n >= front_pad_n and curr_n < N);
            uint64_t read_noc_addr = get_noc_addr(i_stick, s);

            // fill l1_write_addr with pad values
#if (not_pad_by_zero)
            noc_async_read(pad_val_noc_addr, l1_write_addr, stick_size_padded);
            noc_async_read_barrier();
#else
            noc_async_read(zeros_noc_addr, l1_write_addr, stick_size_padded);
            noc_async_read_barrier();
#endif
            DPRINT << "check memory right after read it to l1 write addr" << ENDL();
            print_pages(l1_write_addr, (stick_size_padded) / sizeof(uint16_t), 1);
            if (read_stick) {
                DPRINT << "into read stick with stick_size_padded_front = " << stick_size_padded_front << ENDL();
#if (front_padding)
                // Read noc into cb_pad_align l1
                noc_async_read(read_noc_addr, get_write_ptr(cb_pad_align), stick_size_bytes);
                noc_async_read_barrier();
                DPRINT << "check memory right after read it to pad align write addr" << ENDL();
                print_pages(get_read_ptr(cb_pad_align), (stick_size_bytes) / sizeof(uint16_t), 1);
                // noc_async_read(pad_align_noc_addr, l1_write_addr, stick_size_bytes);
                // noc_async_read_barrier();

                // move data from cb_pad_align to l1_write_addr
                memmove(
                    (void*)(l1_write_addr + stick_size_padded_front),
                    (void*)(get_read_ptr(cb_pad_align)),
                    (size_t)(stick_size_bytes));
                // tt::data_movement::common::tt_memmove<false, true, false, stick_size_padded>(pad_align_noc_addr,
                // l1_write_addr, stick_size_padded_front); noc_async_read(zeros_noc_addr, l1_write_addr,
                // stick_size_padded_front);
#else
                noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
#endif
                /* #if (not_pad_by_zero)
                                if constexpr (stick_size_padded_front != 0) {
                                    for (uint32_t j = 0; j < num_front_pad_sticks_read; ++j) {
                                        noc_async_read(pad_val_noc_addr, l1_write_addr, row_major_min_bytes);
                                        l1_write_addr += row_major_min_bytes;
                                    }
                                }
                #else
                                if constexpr (stick_size_padded_front != 0) {
                                    noc_async_read(zeros_noc_addr, l1_write_addr, stick_size_padded_front);
                                    l1_write_addr += stick_size_padded_front;
                                }
                #endif

                                noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
                                l1_write_addr += stick_size_bytes;
                                i_stick++;

                #if (not_pad_by_zero)
                                if constexpr (stick_size_padded_end != 0) {
                                    for (uint32_t j = 0; j < num_end_pad_sticks_read; ++j) {
                                        noc_async_read(pad_val_noc_addr, l1_write_addr, row_major_min_bytes);
                                        l1_write_addr += row_major_min_bytes;
                                    }
                                }
                #else
                                if constexpr (stick_size_padded_end != 0) {
                                    noc_async_read(zeros_noc_addr, l1_write_addr, stick_size_padded_end);
                                    l1_write_addr += stick_size_padded_end;
                                }
                #endif */

            } else {
                DPRINT << "into read padded scratch" << ENDL();
                /* #if (not_pad_by_zero)
                                for (uint32_t j = 0; j < num_sticks_padded_read; ++j) {
                                    noc_async_read(pad_val_noc_addr, l1_write_addr, row_major_min_bytes);
                                    l1_write_addr += row_major_min_bytes;
                                }
                #else
                                for (uint32_t j = 0; j < num_zero_pad_sticks_read; ++j) {
                                    auto read_bytes = j == num_zero_pad_sticks_read - 1 ? last_zero_stick_size : 512;
                                    noc_async_read(zeros_noc_addr, l1_write_addr, read_bytes);
                                    l1_write_addr += read_bytes;
                                }
                #endif */
            }

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
        DPRINT << "number of elements in a page = " << stick_size_padded / sizeof(uint16_t) << ENDL();
        print_pages(get_read_ptr(cb_in0), stick_size_padded / sizeof(uint16_t), 1);
        cb_push_back(cb_in0, num_read_per_barrier);
    }
}
