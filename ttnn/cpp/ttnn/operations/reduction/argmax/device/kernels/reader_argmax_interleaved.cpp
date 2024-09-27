// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

//#include "debug/dprint.h"

// Optimized function to compare two bfloat16 values using integer arithmetic
bool bfloat16_greater(uint16_t bf16_a, uint16_t bf16_b) {
    /*
    bfloat16 format (16 bits total):
    [Sign (1 bit)][Exponent (8 bits)][Mantissa (7 bits)]
       bit 15         bits 14-7          bits 6-0

    Comparison Logic:
    - If signs differ:
        - If bf16_a is positive (sign bit 0), it is greater.
        - If bf16_a is negative (sign bit 1), it is not greater.
    - If signs are the same:
        - Positive numbers: higher bits mean greater value.
        - Negative numbers: higher bits mean smaller value (reverse comparison).
    */

    // Check if signs are different
    if ((bf16_a ^ bf16_b) & 0x8000) {
        // Signs differ: if bf16_a is positive, it's greater
        return (bf16_a & 0x8000) == 0;
    }

    // Signs are the same
    if (bf16_a & 0x8000) {
        // Both negative: reverse comparison
        return bf16_a < bf16_b;
    } else {
        // Both positive: regular comparison
        return bf16_a > bf16_b;
    }
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_intermed0 = get_compile_time_arg_val(1);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr uint32_t in0_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(5);
    constexpr uint32_t B = get_compile_time_arg_val(6);
    constexpr uint32_t C = get_compile_time_arg_val(7);
    constexpr uint32_t H = get_compile_time_arg_val(8);
    constexpr uint32_t W = get_compile_time_arg_val(9);
    constexpr uint32_t dim = get_compile_time_arg_val(10);
    constexpr uint32_t all = get_compile_time_arg_val(11);

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = in0_stick_size};

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};

    // Use cb as L1 scratch memory
    uint32_t out_addr = get_write_ptr(cb_id_intermed0);
    volatile tt_l1_ptr uint32_t* max_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

    // Use cb as L1 scratch memory
    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    uint32_t max_index = 0;
    uint32_t max_val = 0;
    uint32_t index_counter = 0;

    for(uint32_t l = 0; l < B; l ++) {
        for(uint32_t k = 0; k < C; k++) {
            for(uint32_t j = 0; j < H; j++) {
                noc_async_read_page(l*C*H + k*H + j, s0, cb_addr);
                noc_async_read_barrier();
                if (dim == 3) {
                    index_counter = 0;
                    max_index = 0;
                    max_val = stick[0];
                }
                for(uint32_t i = 0; i < W; i++) {
                    uint16_t val = stick[i];
                    if(bfloat16_greater(val, max_val)) {
                        max_index = index_counter;
                        max_val = val;
                    }
                    index_counter++;

                }
                if (dim == 3) {
                    max_vals[l*C*H + k*H + j] = max_index;
                }
            }
        }
    }
    // TODO: Generalize write for argmax for other dims
    if  constexpr (all) {
        max_vals[0] = max_index;
    }
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);

    noc_async_write(out_addr, dst_noc_addr, out_stick_size);
    noc_async_write_barrier();
}
