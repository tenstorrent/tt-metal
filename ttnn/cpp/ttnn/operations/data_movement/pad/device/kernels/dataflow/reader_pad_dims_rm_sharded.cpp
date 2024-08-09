// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"



inline __attribute__((always_inline))
void fill_pad_cb_with_val(const uint32_t cb_id, const uint32_t num_bytes, const uint32_t val) {

    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_bytes / 2; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {

    uint32_t read_pad_only = get_arg_val<uint32_t>(0);
    uint32_t read_noc_x = get_arg_val<uint32_t>(1);
    uint32_t read_noc_y  = get_arg_val<uint32_t>(2);

    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t W_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t H_padded = get_compile_time_arg_val(2);
    constexpr uint32_t W_padded_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t rem_W_padded_size_bytes = get_compile_time_arg_val(4);

    #define not_pad_by_zero get_compile_time_arg_val(5) == 1
    constexpr uint32_t packed_pad_value = get_compile_time_arg_val(6);
    constexpr uint32_t row_major_min_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t num_rem_sticks_read = get_compile_time_arg_val(8);
    constexpr uint32_t num_sticks_padded_read = get_compile_time_arg_val(9);

    constexpr bool HW_has_no_padding = (W_size_bytes == W_padded_size_bytes && H == H_padded);



    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_pad = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;

    const uint32_t stick_size_bytes = W_size_bytes;
    const uint32_t rem_stick_size_bytes = rem_W_padded_size_bytes;


    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint64_t pad_val_addr = get_read_ptr(cb_pad);
    uint64_t pad_val_noc_addr = get_noc_addr(pad_val_addr);

    #if (not_pad_by_zero)
        fill_pad_cb_with_val(cb_pad, row_major_min_bytes, packed_pad_value);
    #endif

    uint32_t l1_read_addr = get_write_ptr(cb_in0);
    uint64_t read_noc_addr = get_noc_addr(read_noc_x, read_noc_y, l1_read_addr);
    uint32_t l1_write_addr = get_write_ptr(cb_out0);

    cb_reserve_back(cb_out0, H_padded);

    if (read_pad_only) {
        #if (not_pad_by_zero)
        noc_async_read_one_packet_set_state(pad_val_noc_addr, row_major_min_bytes);
        #else
        noc_async_read_one_packet_set_state(zeros_noc_addr, W_padded_size_bytes);
        #endif

        for (uint32_t iter = 0; iter < H_padded; ++iter) {
            #if (not_pad_by_zero)
            if constexpr(rem_W_padded_size_bytes != 0) {
                for (uint32_t j = 0; j < num_sticks_padded_read; ++j) {
                    noc_async_read_one_packet_with_state(pad_val_noc_addr, l1_write_addr);
                    l1_write_addr += row_major_min_bytes;
                }
            }
            #else
                noc_async_read_one_packet_with_state(zeros_noc_addr, l1_write_addr);
                l1_write_addr += W_padded_size_bytes;
            #endif
        }
    } else {

        if constexpr(HW_has_no_padding) {
            noc_async_read_one_packet_set_state(read_noc_addr, stick_size_bytes);
        }

        for (uint32_t iter = 0; iter < H; ++iter) {
            if constexpr(HW_has_no_padding) {
                noc_async_read_one_packet_with_state(read_noc_addr, l1_write_addr);
            } else {
                noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
            }
            read_noc_addr += stick_size_bytes;
            l1_write_addr += stick_size_bytes;

            #if (not_pad_by_zero)
            if constexpr(rem_W_padded_size_bytes != 0) {
                for (uint32_t j = 0; j < num_rem_sticks_read; ++j) {
                    noc_async_read(pad_val_noc_addr, l1_write_addr, row_major_min_bytes);
                    l1_write_addr += row_major_min_bytes;
                }
            }
            #else
            if constexpr(rem_W_padded_size_bytes != 0) {
                noc_async_read(zeros_noc_addr, l1_write_addr, rem_stick_size_bytes);
                l1_write_addr += rem_stick_size_bytes;
            }
            #endif
        }

        for (uint32_t iter = 0; iter < H_padded - H; ++iter) {
            #if (not_pad_by_zero)
            if constexpr(rem_W_padded_size_bytes != 0) {
                for (uint32_t j = 0; j < num_sticks_padded_read; ++j) {
                    noc_async_read(pad_val_noc_addr, l1_write_addr, row_major_min_bytes);
                    l1_write_addr += row_major_min_bytes;
                }
            }
            #else
                noc_async_read(zeros_noc_addr, l1_write_addr, W_padded_size_bytes);
                l1_write_addr += W_padded_size_bytes;
            #endif
        }
    }

    noc_async_read_barrier();
    cb_push_back(cb_out0, H_padded);

}
