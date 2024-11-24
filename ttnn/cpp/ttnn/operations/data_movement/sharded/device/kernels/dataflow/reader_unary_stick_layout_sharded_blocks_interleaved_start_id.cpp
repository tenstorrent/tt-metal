// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#define ENABLE_DEBUG 1

#if ENABLE_DEBUG
#include "debug/dprint.h"

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++ page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif

void kernel_main() {
    DPRINT << "Entering i2s reader kernel!" << ENDL();
    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t stick_size               = get_arg_val<uint32_t>(1);
    const uint32_t block_height             = get_arg_val<uint32_t>(2);
    const uint32_t block_width_bytes        = get_arg_val<uint32_t>(3);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(4);
    const bool aligned                      = static_cast<bool>(get_arg_val<uint32_t>(5));
    const uint32_t aligned_input_width_offset_bytes = get_arg_val<uint32_t>(6);
    const uint32_t aligned_block_width_bytes = get_arg_val<uint32_t>(7);
    const uint32_t aligned_offset           = get_arg_val<uint32_t>(8);
    const uint32_t start_id                 = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);

    constexpr bool src0_is_dram          = get_compile_time_arg_val(2) == 1;
    #define src_stick_size_is_pow2 get_compile_time_arg_val(3) == 1
    #if (src_stick_size_is_pow2)
    constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(4);
    const InterleavedPow2AddrGen<src0_is_dram> s0 = {
        .bank_base_address = src_addr + aligned_input_width_offset_bytes,
        .log_base_2_of_page_size = src_log_base_2_of_page_size
    };
    #else
    const InterleavedAddrGen<src0_is_dram> s0 = {
        .bank_base_address = src_addr + aligned_input_width_offset_bytes,
        .page_size = stick_size
    };
    #endif
    uint32_t stick_id = start_id;
    cb_reserve_back(cb_id_in0, block_height);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    DPRINT << "I2S READER OUTPUT" << ENDL();

    DPRINT << "stick_size: " << stick_size << ENDL();

    DPRINT << "block_height" << block_height << ENDL();
    DPRINT << "block_width_bytes: " << block_width_bytes << ENDL();
    DPRINT << "padded_block_width_bytes: " << padded_block_width_bytes << ENDL();


    DPRINT << "aligned_input_width_offset_bytes: " << aligned_input_width_offset_bytes << ENDL();
    DPRINT << "aligned_block_width_bytes: " << aligned_block_width_bytes << ENDL();
    DPRINT << "aligned_offset: " << aligned_offset << ENDL();
    //hack the code snippet here.
    if (aligned) {
        DPRINT << "Aligned Case:" << ENDL();
        for (uint32_t h = 0; h < block_height; ++h) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s0);

            //uint32_t scratch_l1_write_addr = get_write_ptr(cb_id_in1);
            //uint64_t scratch_l1_noc_read_addr = get_noc_addr(scratch_l1_write_addr + aligned_offset);
            /*
            uint64_t scratch_l1_noc_read_addr = get_noc_addr(scratch_l1_write_addr);
            noc_async_read(src_noc_addr, scratch_l1_write_addr, block_width_bytes);
            noc_async_read_barrier();
            noc_async_read(scratch_l1_noc_read_addr, l1_write_addr, block_width_bytes);
            */

            noc_async_read(src_noc_addr, l1_write_addr, block_width_bytes);
            DPRINT << "Print single page after noc->l1 transfer." << ENDL();
            print_pages(l1_write_addr, block_width_bytes / sizeof(uint16_t), 1);
            stick_id++;
            l1_write_addr += block_width_bytes;
        }
    } else {
        DPRINT << "Unaligned Case:" << ENDL();

        cb_reserve_back(cb_id_in1, 1);
        uint32_t scratch_l1_write_addr = get_write_ptr(cb_id_in1);
        //uint64_t scratch_l1_noc_read_addr = get_noc_addr(scratch_l1_write_addr + aligned_offset);
        uint64_t scratch_l1_noc_read_addr = get_noc_addr(scratch_l1_write_addr);
        for (uint32_t h = 0; h < block_height; ++h) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
            noc_async_read(src_noc_addr, scratch_l1_write_addr, aligned_block_width_bytes);
            DPRINT << "Print single page after noc->scratch l1 transfer." << ENDL();
            print_pages(scratch_l1_write_addr, block_width_bytes / sizeof(uint16_t), 1);
            noc_async_read_barrier();
            noc_async_read(scratch_l1_noc_read_addr, l1_write_addr, block_width_bytes);
            stick_id++;
            l1_write_addr += block_width_bytes;
        }
    }
    DPRINT << "out of if clause" << ENDL();
    noc_async_read_barrier();



    print_pages(get_read_ptr(cb_id_in0), block_width_bytes / sizeof(uint16_t), block_height);

    cb_push_back(cb_id_in0, block_height);
}
