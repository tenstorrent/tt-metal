// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// #include "debug/dprint.h"

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

    //cb_reserve_back(cb_id_intermed0, C*H*W);
    //uint32_t indicies_addr = get_write_ptr(cb_id_intermed0);
    //volatile tt_l1_ptr uint32_t* max_indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    uint32_t max_index = 0;
    uint32_t max_val = 0;
    uint32_t index_counter = 0;
    for(uint32_t l = 0; l < B; l ++) {
        for(uint32_t k = 0; k < C; k++) {
            for(uint32_t j = 0; j < H; j++) {
                // load stick
                // DPRINT << (l*C*H + k*H + j) << ENDL();
                noc_async_read_page(l*C*H + k*H + j, s0, cb_addr);
                noc_async_read_barrier();
                for(uint32_t i = 0; i < W; i++) {
                    if constexpr (all) {
                        uint16_t val = stick[i];
                        if(bfloat16_greater(val, max_val)) {
                            // DPRINT << "new max " << HEX() << (val) << "\nGT old max " << (max_val) << ENDL();
                            // DPRINT << "new idx " << DEC() << (index_counter) << "\nGT old idx " << (max_index) << ENDL();
                            // DPRINT << DEC() << (max_index) << ENDL();
                            max_index = index_counter;
                            max_val = val;
                        }
                        // DPRINT << "[" << index_counter << "] = " << HEX() << (val) << ENDL();
                        index_counter++;
                    }
                    else {
                    /*
                        if(dim == 3) {
                            if(bfloat16_greater(bfloat16_max_vals[l][k][j] < stick[i]) {
                                bfloat16_max_vals[l][k][j] = stick[i];
                                max_indices[l][k][j] = i;
                            }
                        }
                        else if(dim == 2) {
                            if(bfloat16_max_vals[l][k][i] < stick[i]) {
                                bfloat16_max_vals[l][k][i] = stick[i];
                                max_indices[l][k][i] = j;
                            }
                        }
                        else if(dim == 1) {
                            if(bfloat16_max_vals[l][j][i] < stick[i]) {
                                bfloat16_max_vals[l][j][i] = stick[i];
                                max_indices[l][j][i] = k;
                            }
                        }
                        else if(dim == 0) {
                            if(bfloat16_greater(stick[i], bfloat16_max_vals[k][j][i])) {
                                bfloat16_max_vals[k][j][i] = stick[i];
                                max_indices[k][j][i] = l;
                            }
                        }
                    */
                    }
                }
            }
        }
    }

    // TODO: Generalize write for argmax for other dims
    max_vals[0] = max_index;
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);
    noc_async_write(out_addr, dst_noc_addr, out_stick_size);
    noc_async_write_barrier();
}
