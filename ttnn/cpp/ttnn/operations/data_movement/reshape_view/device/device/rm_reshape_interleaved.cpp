// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


/*
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
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    //We are guranteed to be in 2D going to 2D

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    //If DDR this is source_page_size_bytes + 64 (rounded up to next 64B), if L1 this is source_page_size_bytes + 16 (rounded up to next 16B)
    const uint32_t source_read_size_bytes = get_arg_val<uint32_t>(2);
    const uint32_t read_start_page = get_arg_val<uint32_t>(3);
    const uint32_t read_end_page = get_arg_val<uint32_t>(4);
    const uint32_t write_start_page = get_arg_val<uint32_t>(5);
    const uint32_t write_start_offset = get_arg_val<uint32_t>(6);
    const uint32_t nop = get_arg_val<uint32_t>(9);

    constexpr bool tensor_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src_aligned_to_64 = get_compile_time_arg_val(1) == 1;
    constexpr bool src_aligned_to_16 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(4);
    constexpr uint32_t source_page_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t dest_page_size_bytes = get_compile_time_arg_val(6);
    constexpr bool source_page_is_pow_2 = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t source_page_pow_2 = get_compile_time_arg_val(8);
    constexpr bool dest_page_is_pow_2 = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t dest_page_pow_2 = get_compile_time_arg_val(10);

    //Since we need to operate on a grid of cores but sometimes pages don't split properly, if nop then don't use this core
    if (nop == 1)
    {
        return;
    }

    const auto s = get_interleaved_addr_gen<tensor_is_dram, source_page_is_pow_2>(
        src_addr, source_page_size_bytes, source_page_pow_2);
    const auto d =
        get_interleaved_addr_gen<tensor_is_dram, dest_page_is_pow_2>(dst_addr, dest_page_size_bytes, dest_page_pow_2);

    uint32_t read_offset = 0;
    uint32_t write_page = write_start_page;
    uint32_t readable = 0;
    uint32_t end_to_write = 0;
    uint32_t transaction = 0;
    uint32_t writable = dest_page_size_bytes - write_start_offset;
    //cb_id_in0 is a CB source_read_size_bytes page size, 1 page
    //cb_id_in1 is a CB dest_page_size_bytes + allignment_to_64 page size, 1 page
    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);
    const uint32_t source_buffer = get_write_ptr(cb_id_in0);
    const uint32_t dest_buffer = get_write_ptr(cb_id_in1);
    cb_push_back(cb_id_in1, 1);
    cb_push_back(cb_id_in0, 1);

    uint64_t dst_noc_addr = get_noc_addr(write_page, d);
    uint64_t write_offset = (dst_noc_addr & OFFSET_16) + write_start_offset;
    uint64_t begin_write_offset = write_offset;
    constexpr bool can_be_clean = ((source_page_size_bytes % 16) == 0 && (dest_page_size_bytes % 16) == 0);
    uint64_t dst_noc_addr_offset = 0;
    for (uint32_t i = read_start_page; i < read_end_page; i++) {
        //Read from source
        uint64_t src_noc_addr = s.get_noc_addr(i,0);

        if constexpr (src_aligned_to_64 || ((!tensor_is_dram) && src_aligned_to_16)) {  // Aligned to 64 bytes or 16
                                                                                        // bytes but L1
            tt::data_movement::common::enhanced_noc_async_read<source_page_size_bytes, false>(
                src_noc_addr, source_buffer, source_page_size_bytes);
            read_offset = 0;
        } else if constexpr (tensor_is_dram) {  // DDR but not alligned to 64 (potentially also not alligned to 16)
            tt::data_movement::common::enhanced_noc_async_read<(source_page_size_bytes + 128), false>(
                src_noc_addr & MASK_64, source_buffer, source_read_size_bytes);
            read_offset = src_noc_addr & OFFSET_64;
        } else {  // L1 but not alligned to 16
            tt::data_movement::common::enhanced_noc_async_read<(source_page_size_bytes + 128), false>(
                src_noc_addr & MASK_16, source_buffer, source_read_size_bytes);
            read_offset = src_noc_addr & OFFSET_16;
        }

        readable = source_page_size_bytes;
        noc_async_read_barrier();

        //Write to dest
        while (readable > 0)
        {
            noc_async_write_barrier();
            if (readable < writable)
            {
                if constexpr (can_be_clean) {
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        source_buffer + read_offset, dst_noc_addr + dst_noc_addr_offset, readable);
                    dst_noc_addr_offset = dst_noc_addr_offset + readable;
                } else {
                    tt::data_movement::common::tt_memmove<false, true, false, dest_page_size_bytes>(
                        dest_buffer + write_offset, source_buffer + read_offset, readable);
                    if (i == read_end_page - 1) {
                        tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                            dest_buffer + begin_write_offset, dst_noc_addr, end_to_write);
                        noc_async_write_barrier();
                        return;
                    }
                    write_offset = write_offset + readable;
                    end_to_write = end_to_write + readable;
                }
                writable = writable - readable;
                readable = 0;

            }
            else if (readable == writable)
            {
                if constexpr (can_be_clean) {
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        source_buffer + read_offset, dst_noc_addr + dst_noc_addr_offset, readable);
                } else {
                    tt::data_movement::common::tt_memmove<false, false, false, dest_page_size_bytes>(
                        dest_buffer + write_offset, source_buffer + read_offset, readable);
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        dest_buffer + begin_write_offset, dst_noc_addr, dest_page_size_bytes);
                }
                dst_noc_addr_offset = 0;

                writable = dest_page_size_bytes;
                readable = 0;
                if (i == read_end_page - 1) {
                    noc_async_write_barrier();
                    return;
                }
                write_page++;
                dst_noc_addr = get_noc_addr(write_page, d);
                if constexpr (!can_be_clean) {
                    end_to_write = 0;
                    write_offset = dst_noc_addr & OFFSET_16;
                    begin_write_offset = write_offset;
                }
            }
            else
            {
                if constexpr (can_be_clean) {
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        source_buffer + read_offset, dst_noc_addr + dst_noc_addr_offset, writable);
                } else {
                    tt::data_movement::common::tt_memmove<false, false, false, dest_page_size_bytes>(
                        dest_buffer + write_offset, source_buffer + read_offset, writable);
                    tt::data_movement::common::enhanced_noc_async_write<dest_page_size_bytes, false>(
                        dest_buffer + begin_write_offset, dst_noc_addr, dest_page_size_bytes);
                }
                // writable < readable
                readable = readable - writable;
                read_offset = read_offset + writable;
                write_page++;
                dst_noc_addr_offset = 0;
                dst_noc_addr = get_noc_addr(write_page, d);
                if constexpr (!can_be_clean) {
                    end_to_write = 0;
                    write_offset = dst_noc_addr & OFFSET_16;
                    begin_write_offset = write_offset;
                }
                writable = dest_page_size_bytes;
            }
        }
    }
    noc_async_write_barrier();
    return;
}
