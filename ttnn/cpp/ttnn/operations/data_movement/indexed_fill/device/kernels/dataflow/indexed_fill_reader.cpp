// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {


    uint32_t batch_ids_addr            = get_arg_val<uint32_t>(0);
    uint32_t batch_id_size             = get_arg_val<uint32_t>(1);
    uint32_t input_addr_a                = get_arg_val<uint32_t>(2);
    uint32_t input_addr_b                = get_arg_val<uint32_t>(3);
    uint32_t stick_size             = get_arg_val<uint32_t>(4);
    uint32_t batch_size_in_sticks              = get_arg_val<uint32_t>(5);
    uint32_t my_batch_id              = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t batch_cb_id = get_compile_time_arg_val(1);
    constexpr bool batch_ids_is_dram          = get_compile_time_arg_val(2) == 1;
    constexpr bool src0_is_dram          = get_compile_time_arg_val(3) == 1;
    constexpr bool src1_is_dram          = get_compile_time_arg_val(4) == 1;
    #define src_stick_size_is_pow2 get_compile_time_arg_val(5) == 1
    #if (src_stick_size_is_pow2)
    constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(6);
    const InterleavedPow2AddrGen<src0_is_dram> s0 = {
        .bank_base_address = input_addr_a,
        .log_base_2_of_page_size = src_log_base_2_of_page_size // TODO(AP): refactor
    };
    const InterleavedPow2AddrGen<src1_is_dram> s1 = {
        .bank_base_address = input_addr_b,
        .log_base_2_of_page_size = src_log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<src0_is_dram> s0 = {
        .bank_base_address = input_addr_a,
        .page_size = stick_size
    };
    const InterleavedAddrGen<src1_is_dram> s1 = {
        .bank_base_address = input_addr_b,
        .page_size = stick_size
    };
    #endif

    const InterleavedAddrGen<batch_ids_is_dram> batchAddr = {
        .bank_base_address = batch_ids_addr,
        .page_size =  batch_id_size << 2
    };


    bool replace_batch = false;
    uint32_t batch_to_replace_id = 0;
    //first go through batch id


    volatile tt_l1_ptr int* addr_ptr;

    if(batch_id_size > 0) {
        uint64_t src_noc_addr = get_noc_addr(0, batchAddr);
        uint32_t l1_write_addr = get_write_ptr(batch_cb_id);
        noc_async_read(src_noc_addr, l1_write_addr, (batch_id_size << 2) );
        noc_async_read_barrier();
        addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(l1_write_addr);
    }
    for (uint32_t i=0; i<batch_id_size; i++) {
        uint32_t batch_id_to_replace = addr_ptr[i];
        if(batch_id_to_replace == my_batch_id) {
            replace_batch = true;
            batch_to_replace_id = i;
        }
    }


    uint32_t start_id;
    if (replace_batch) {
        start_id = batch_to_replace_id;
    } else {
        start_id = my_batch_id;
    }



    uint32_t end_id = start_id + batch_size_in_sticks;
    for (uint32_t i = start_id; i < end_id; ++ i) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint64_t src_noc_addr;
        if(replace_batch) {
            src_noc_addr = get_noc_addr(i, s1);
        }
        else {
            src_noc_addr = get_noc_addr(i, s0);
        }
        noc_async_read(src_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    }
}
