// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
NOTE: This function is an improvement on rm_reshape_interleaved.cpp but it has a bug causing a hang for some cases that
needs to be debugged first

Function reads from RM and writes to RM

Assumptions:

Compile arguments
0. src0_is_dram: 1 if source is dram else 0
1. read_size_is_pow2: 1 if read size is power of 2 else 0
2. log_base_2_of_page_size: log base 2 of page size
3. write_size_is_pow2: 1 if write size is power of 2 else 0
4. log_base_2_of_page_size: log base 2 of page size
5. needs_read_allignment: 1 if read needs allignment else 0
//Needed if BRAM and page size is not multiple of 64 bytes

Runtime arguments
0. src_addr: source address
1. dst_addr: destination address
2. source_page_size_bytes: source page size in bytes
3. dest_page_size_bytes: destination page size in bytes
4. source_read_size_bytes: source read size in bytes
5. read_start_page: read start page
6. read_end_page: read end page
7. write_start_page: write start page
*/
#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // We are guranteed to be in 2D going to 2D

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t source_page_size_bytes = get_arg_val<uint32_t>(2);
    const uint32_t dest_page_size_bytes = get_arg_val<uint32_t>(3);
    // If DDR this is source_page_size_bytes + 64 (rounded up to next 64B), if L1 this is source_page_size_bytes + 16
    // (rounded up to next 16B)
    const uint32_t source_read_size_bytes = get_arg_val<uint32_t>(4);
    const uint32_t read_start_page = get_arg_val<uint32_t>(5);
    const uint32_t read_end_page = get_arg_val<uint32_t>(6);
    const uint32_t write_start_page = get_arg_val<uint32_t>(7);
    const uint32_t write_start_offset = get_arg_val<uint32_t>(8);
    const uint32_t nop = get_arg_val<uint32_t>(9);
    const uint64_t ping_read_has_data = get_noc_addr(get_semaphore(get_arg_val<uint32_t>(10)));
    const uint64_t pong_read_has_data = get_noc_addr(get_semaphore(get_arg_val<uint32_t>(11)));
    volatile uint32_t* ping_buf_is_free =
        reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(12)));
    volatile uint32_t* pong_buf_is_free =
        reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(13)));
    constexpr bool tensor_is_dram = get_compile_time_arg_val(0) == 1;
#define src_aligned_to_64 get_compile_time_arg_val(1) == 1
#define src_aligned_to_16 get_compile_time_arg_val(2) == 1
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(4);
    // Since we need to operate on a grid of cores but sometimes pages don't split properly, if nop then don't use this
    // core
    if (nop == 1) {
        return;
    }

    const InterleavedAddrGen<tensor_is_dram> s = {.bank_base_address = src_addr, .page_size = source_page_size_bytes};

    uint32_t read_offset = 0;
    uint32_t write_page = write_start_page;
    uint32_t readable = 0;
    uint32_t end_to_write = 0;
    uint32_t transaction = 0;
    uint32_t writable = dest_page_size_bytes - write_start_offset;
    // cb_id_in0 is a CB source_read_size_bytes +4 page size, 1 page
    // cb_id_in1 is a CB source_read_size_bytes +4 page size, 1 page
    // cb_id_in1 is a CB dest_page_size_bytes + allignment_to_64 page size, 1 page
    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);
    cb_reserve_back(cb_id_in2, 1);
    const uint32_t source_buffer_ping = get_write_ptr(cb_id_in0);
    const uint32_t source_buffer_pong = get_write_ptr(cb_id_in1);
    const uint32_t dest_buffer = get_write_ptr(cb_id_in2);
    cb_push_back(cb_id_in0, 1);
    cb_push_back(cb_id_in1, 1);
    cb_push_back(cb_id_in2, 1);
    uint32_t source_buffer;

    volatile tt_l1_ptr std::uint32_t* read_offset_ptr_ping =
        (volatile tt_l1_ptr uint32_t*)(source_buffer_ping + source_read_size_bytes);
    volatile tt_l1_ptr std::uint32_t* read_offset_ptr_pong =
        (volatile tt_l1_ptr uint32_t*)(source_buffer_pong + source_read_size_bytes);
    bool is_ping = true;
    bool first = true;
    bool second = true;
    bool third = true;
    bool first_pong = true;
    bool second_pong = true;
    bool third_pong = true;
    for (uint32_t i = read_start_page; i < read_end_page; i++) {
        // Read from source
        uint64_t src_noc_addr = s.get_noc_addr(i, 0);
        if (is_ping) {
            if (first) {
                first = false;
                WAYPOINT("FARW");
            } else if (second) {
                second = false;
                WAYPOINT("SARW");
            } else if (third) {
                third = false;
                WAYPOINT("TARW");
            } else {
                WAYPOINT("ARW");
            }
            source_buffer = source_buffer_ping;
            noc_semaphore_wait(ping_buf_is_free, 1);
            WAYPOINT("ARD");
        } else {
            if (first_pong) {
                first_pong = false;
                WAYPOINT("FBRW");
            } else if (second_pong) {
                second_pong = false;
                WAYPOINT("SBRW");
            } else {
                WAYPOINT("BRW");
            }
            source_buffer = source_buffer_pong;
            noc_semaphore_wait(pong_buf_is_free, 1);
            WAYPOINT("BRD");
        }

#if (src_aligned_to_64 || ((!tensor_is_dram) && src_aligned_to_16))
        // Aligned to 64 bytes or 16 bytes but L1
        noc_async_read(src_noc_addr, source_buffer, source_page_size_bytes);
        read_offset = 0;
#elif (tensor_is_dram)
        // DDR but not alligned to 64 (potentially also not alligned to 16)
        noc_async_read(src_noc_addr & MASK_64, source_buffer, source_read_size_bytes);
        read_offset = src_noc_addr & OFFSET_64;
#else
        // L1 but not alligned to 16
        noc_async_read(src_noc_addr & MASK_16, source_buffer, source_read_size_bytes);
        read_offset = src_noc_addr & OFFSET_16;
#endif
        if (is_ping) {
            *read_offset_ptr_ping = read_offset;
        } else {
            *read_offset_ptr_pong = read_offset;
        }
        noc_async_read_barrier();
        if (is_ping) {
            noc_semaphore_inc(ping_read_has_data, 1);
        } else {
            noc_semaphore_inc(pong_read_has_data, 1);
        }
    }
    return;
}
