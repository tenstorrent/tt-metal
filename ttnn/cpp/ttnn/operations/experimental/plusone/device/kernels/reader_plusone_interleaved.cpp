// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "debug/dprint.h"

uint16_t float_add_one(uint16_t bf16_a) {
    // Extract components
    uint16_t sign = (bf16_a >> 15) & 0x1;  // sign bit
    uint16_t exp = (bf16_a >> 7) & 0xFF;   // exponent (8 bits)
    uint16_t man = bf16_a & 0x7F;          // mantissa (7 bits)

    // Handle special cases: NaN or infinity (exponent is all 1s)
    if (exp == 0xFF) {
        // If it's NaN or infinity, return the same value
        return bf16_a;
    }

    // If the exponent is zero, the value is either subnormal or zero
    if (exp == 0) {
        if (man == 0) {
            // If it's exactly zero, return the smallest positive normal number
            return (1 << 7); // 1.0 in bfloat16 representation
        } else {
            // It's a subnormal number, increment the mantissa
            man++;
            if (man == 0x80) {
                // Overflow in mantissa, increment exponent
                man = 0;
                exp = 1;
            }
        }
    } else {
        // Normal number: increment mantissa
        man++;
        if (man == 0x80) {
            // If the mantissa overflows, increment the exponent and reset mantissa
            man = 0;
            exp++;
            if (exp == 0xFF) {
                // If exponent overflows to 0xFF, it's infinity
                man = 0;
            }
        }
    }

    // Reconstruct the bfloat16 number with the new values
    return (sign << 15) | (exp << 7) | man;
}


void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = stick_size};

    // Use cb as L1 scratch memory
    uint32_t cb_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);


    noc_async_read_page(0, s0, cb_addr);
    noc_async_read_barrier();
    for(uint32_t i = 0; i < W; i++) {
        uint32_t val = stick[i];
        stick[i] = val+1;
        //DPRINT << "val: " << val << ENDL();
    }

    uint64_t dst_noc_addr = get_noc_addr(0, s0);

    noc_async_write(cb_addr, dst_noc_addr, stick_size);
    noc_async_write_barrier();
}
