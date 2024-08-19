// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"



inline __attribute__((always_inline))
void fill_pad_cb_with_val(const uint32_t cb_id, const uint32_t num_bytes_risc, uint32_t num_noc_transfer, const uint32_t val) {

    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_bytes_risc / 2; ++i) {
        ptr[i] = val;
    }

    uint32_t pad_val_addr = get_read_ptr(cb_id);
    uint64_t pad_val_noc_addr = get_noc_addr(pad_val_addr);
    uint32_t l1_write_addr = pad_val_addr + num_bytes_risc;

    for (uint32_t i = 0; i < num_noc_transfer; ++i) {
        noc_async_read(pad_val_noc_addr, l1_write_addr, num_bytes_risc);
        l1_write_addr += num_bytes_risc;
    }
    noc_async_read_barrier();
}

void kernel_main() {

    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(0);
    uint32_t start_id  = get_arg_val<uint32_t>(1);
    tt_l1_ptr uint32_t * start_dim_offset = (tt_l1_ptr uint32_t*)(get_arg_addr(2));

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t N_padded = get_compile_time_arg_val(4);
    constexpr uint32_t H_padded = get_compile_time_arg_val(5);
    constexpr uint32_t C_padded = get_compile_time_arg_val(6);

    #define not_pad_by_zero get_compile_time_arg_val(7) == 1
    #if (not_pad_by_zero)
    constexpr uint32_t packed_pad_value = get_compile_time_arg_val(8);
    constexpr uint32_t row_major_min_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t num_sticks_padded_read = get_compile_time_arg_val(10);
    #endif


    constexpr auto cb_pad = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint32_t pad_val_addr = get_read_ptr(cb_pad);
    uint64_t pad_val_noc_addr = get_noc_addr(pad_val_addr);

    #if (not_pad_by_zero)
        fill_pad_cb_with_val(cb_pad, row_major_min_bytes, num_sticks_padded_read, packed_pad_value);
    #endif

    uint32_t l1_write_addr = get_write_ptr(cb_out0);

    uint32_t i_stick = start_id;
    uint32_t curr_c = start_dim_offset[2], curr_h = start_dim_offset[1], curr_n = start_dim_offset[3];
    for (uint32_t iter = 0; iter < num_sticks_per_core; ++iter) {
        bool read_stick = curr_h < H and curr_c < C and curr_n < N;

        if (read_stick) {
            l1_write_addr += stick_size_bytes;
            i_stick++;

        } else {

            #if (not_pad_by_zero)
                noc_async_read(pad_val_noc_addr, l1_write_addr, stick_size_bytes);
                l1_write_addr += stick_size_bytes;
            #else
                noc_async_read(zeros_noc_addr, l1_write_addr, stick_size_bytes);
                l1_write_addr += stick_size_bytes;
            #endif
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

}
