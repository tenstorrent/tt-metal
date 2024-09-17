// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

// Function to compare two bfloat16 values using integer arithmetic
bool bfloat16_greater(uint16_t bf16_a, uint16_t bf16_b) {
    // Extract signs
    uint16_t sign_a = (bf16_a >> 15) & 0x1;
    uint16_t sign_b = (bf16_b >> 15) & 0x1;

    uint16_t exp_a = (bf16_a >> 7) & 0xFF;
    uint16_t exp_b = (bf16_b >> 7) & 0xFF;

    uint16_t man_a = bf16_a & 0x7F;
    uint16_t man_b = bf16_b & 0x7F;

    // TODO: Investigate subnormal support
    // uint16_t subnormal_a = (exp_a == 0x00);
    // uint16_t subnormal_b = (exp_b == 0x00);

    // DPRINT << HEX() << (bf16_a) << " > " << bf16_b << ENDL();
    // DPRINT << HEX() << (sign_a) << " signs " << sign_b << ENDL();
    // DPRINT << HEX() << (exp_a) << " exp " << exp_b << ENDL();
    // DPRINT << HEX() << (man_a) << " man " << man_b << ENDL();

    // If signs are different, the one without the sign bit is greater
    if (sign_a != sign_b) {
        // DPRINT << "sign_b > sign_a: " << (int)(sign_b > sign_a) << ENDL();
        return sign_b > sign_a;
    }

    // If signs are the same, compare the exponent and mantissa
    if (sign_a == 0) { // Positive numbers
        if(exp_a == exp_b) {
            // DPRINT << "man_a > man_b: " << (int)(man_a > man_b) << ENDL();
            return man_a > man_b;
        }
        // DPRINT << "exp_a > exp_b: " << (int)(exp_a > exp_b) << ENDL();
        return exp_a > exp_b;
    } else { // Negative numbers
        if(exp_a == exp_b) {
            // DPRINT << "man_a < man_b: " << (int)(man_a < man_b) << ENDL();
            return man_a < man_b;
        }
        // DPRINT << "exp_a < exp_b: " << (int)(exp_a < exp_b) << ENDL();
        return exp_a < exp_b;
    }
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t core_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_intermed0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(2);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr uint32_t in0_stick_size = get_compile_time_arg_val(5);
    constexpr uint32_t intermed0_stick_size = get_compile_time_arg_val(6);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(7);
    constexpr uint32_t B = get_compile_time_arg_val(8);
    constexpr uint32_t C = get_compile_time_arg_val(9);
    constexpr uint32_t H = get_compile_time_arg_val(10);
    constexpr uint32_t W = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores = get_compile_time_arg_val(12);
    uint32_t semaphore_addr_ptr = get_semaphore(get_compile_time_arg_val(13));
    constexpr uint32_t final_cores_physical_x = get_compile_time_arg_val(14);
    constexpr uint32_t final_cores_physical_y = get_compile_time_arg_val(15);
    bool reducer_core = core_id == 0? 1 : 0;

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = in0_stick_size};

    // Use cb as L1 scratch memory
    uint32_t intermed_addr = get_write_ptr(cb_id_intermed0);
    volatile tt_l1_ptr uint32_t* max_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(intermed_addr);

    // Use cb as L1 scratch memory
    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    uint32_t max_index = 0;
    uint32_t max_val = 0;
    uint32_t index_counter = 0;
    uint32_t core_offset = core_id*W;
    uint32_t page_number = core_id/num_cores;

    noc_async_read_page(page_number, s0, cb_addr); // page size is original_W = W*num_cores size
    noc_async_read_barrier();

    index_counter = core_offset;
    max_index = index_counter;
    max_val = stick[max_index];
    for(uint32_t i = core_offset; i < W+core_offset; i++) {
        uint16_t val = stick[i];
        DPRINT << "W"<< i<< ":"<<val << ENDL();
        if(bfloat16_greater(val, max_val)) {
            max_index = index_counter;
            max_val = val;
        }
        index_counter++;

    }

    // set max_vals for reader and writer kernels
    max_vals[core_id%2] = max_index;

    DPRINT << max_vals[0] << " " << max_vals[1] << ENDL();

    // write max_vals to reducer core CB
    uint64_t dst_cb_addr = get_noc_addr(final_cores_physical_x, final_cores_physical_y, intermed_addr);

    noc_async_write(intermed_addr + (core_id%2)*4, dst_cb_addr + core_id*4, 4);
    noc_async_write_barrier();

    // inc noc semaphore
    const uint64_t in0_sender_semaphore_noc_addr = get_noc_addr(final_cores_physical_x, final_cores_physical_y, semaphore_addr_ptr);
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);

    if (reducer_core) {
        // wait for semaphore
        volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr_ptr);
        noc_semaphore_wait_min(in0_receiver_semaphore_addr_ptr, num_cores);

        // Use cb as L1 scratch memory
        uint32_t out_addr = get_write_ptr(cb_id_out0);
        volatile tt_l1_ptr uint32_t* max_vals_final = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);
        // re-use intermed cb
        uint32_t intermed_re_addr = get_write_ptr(cb_id_intermed0);
        volatile tt_l1_ptr uint32_t* max_vals_reduce = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(intermed_re_addr);

        max_index = max_vals_reduce[0];
        max_val = stick[max_index];
        for(uint32_t i = 0; i < num_cores; i++) {
            uint32_t index = max_vals_reduce[i];
            DPRINT << "core"<< i<< ":"<<index << ENDL();
            uint16_t val = index; //stick[index];
            if(bfloat16_greater(val, max_val)) {
                max_index = index;
                max_val = val;
            }
        }
        max_vals_final[0] = max_index;

        const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};
        uint64_t dst_noc_addr = get_noc_addr(0, s_out);
        noc_async_write(out_addr, dst_noc_addr, out_stick_size);
        noc_async_write_barrier();
    }
}
